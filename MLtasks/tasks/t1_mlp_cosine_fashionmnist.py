import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
 
# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# ── Metadata ──────────────────────────────────────────────────────────────────
def get_task_metadata() -> dict:
    return {
        'task_id': 'mlp_cosine_fashionmnist',
        'series': 'Neural Networks (MLP)',
        'algorithm': 'MLP + Cosine Annealing with Linear Warmup',
        'dataset': 'FashionMNIST',
        'metrics': ['accuracy', 'loss', 'mse', 'r2'],
        'threshold': {'val_accuracy': 0.87},
    }
 
# ── Data ──────────────────────────────────────────────────────────────────────
def make_dataloaders(batch_size: int = 256, val_fraction: float = 0.1):
    """Download FashionMNIST and return train/val/test DataLoaders."""
    mean, std = (0.2860,), (0.3530,)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
 
    full_train = torchvision.datasets.FashionMNIST(
        root='/tmp/fashionmnist', train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.FashionMNIST(
        root='/tmp/fashionmnist', train=False, download=True, transform=test_tf)
 
    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
 
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader
 
# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """3-hidden-layer MLP with BatchNorm + Dropout."""
 
    def __init__(self, in_dim: int = 784, hidden: int = 512,
                 n_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))
 
def build_model(device: torch.device) -> nn.Module:
    model = MLP()
    return model.to(device)
 
# ── LR Schedule helpers ───────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup then cosine annealing (pure-Python, no torch.optim.lr_scheduler)."""
 
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 lr_max: float, lr_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epoch = 0
 
    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.warmup:
            lr = self.lr_max * (e / self.warmup)
        else:
            t = e - self.warmup
            T = self.total - self.warmup
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * t / T))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr
 
# ── Train / Evaluate ──────────────────────────────────────────────────────────
def train(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
          criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)
 
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
 
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(X)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_targets.append(y.cpu().numpy())
 
    targets = np.concatenate(all_targets)
    preds   = np.concatenate(all_preds)
    probs   = np.concatenate(all_probs)
 
    # One-hot for MSE/R2
    n_cls = probs.shape[1]
    oh = np.zeros((len(targets), n_cls))
    oh[np.arange(len(targets)), targets] = 1.0
 
    return {
        'loss':     total_loss / len(loader.dataset),
        'accuracy': float(accuracy_score(targets, preds)),
        'mse':      float(mean_squared_error(oh, probs)),
        'r2':       float(r2_score(oh, probs)),
    }
 
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X, _ in loader:
            all_preds.append(model(X.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(all_preds)
 
# ── Artifacts ─────────────────────────────────────────────────────────────────
def save_artifacts(model: nn.Module, history: dict, final_metrics: dict):
    # Model checkpoint
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'mlp_fashionmnist.pt'))
 
    # Metrics JSON
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
 
    # Training curves + LR schedule
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
 
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
 
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
 
    axes[2].plot(history['lr'], color='orange')
    axes[2].set_title('Learning Rate (Warmup + Cosine)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
 
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_cosine_fashionmnist_curves.png'), dpi=100)
    plt.close()
    print(f'Artifacts saved to {OUTPUT_DIR}')
 
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    print('=' * 60)
    print('MLP + Cosine Annealing Warmup  |  FashionMNIST')
    print('=' * 60)
    print(f'Device: {device}')
 
    # Hyper-parameters
    EPOCHS        = 30
    WARMUP_EPOCHS = 5
    LR_MAX        = 3e-3
    LR_MIN        = 1e-6
    BATCH_SIZE    = 256
 
    train_loader, val_loader, test_loader = make_dataloaders(BATCH_SIZE)
    model     = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LR_MAX, LR_MIN)
 
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], lr=[])
 
    for epoch in range(1, EPOCHS + 1):
        lr = scheduler.step()
        t_loss = train(model, train_loader, optimizer, criterion, device)
        t_met  = evaluate(model, train_loader, criterion, device)
        v_met  = evaluate(model, val_loader,   criterion, device)
 
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_met['loss'])
        history['train_acc'].append(t_met['accuracy'])
        history['val_acc'].append(v_met['accuracy'])
        history['lr'].append(lr)
 
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{EPOCHS}  lr={lr:.6f}  '
                  f'train_loss={t_loss:.4f}  val_loss={v_met["loss"]:.4f}  '
                  f'val_acc={v_met["accuracy"]:.4f}')
 
    # ── Final evaluation ──────────────────────────────────────────────────────
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics   = evaluate(model, val_loader,   criterion, device)
    test_metrics  = evaluate(model, test_loader,  criterion, device)
 
    print('\n' + '=' * 60)
    print('Final Metrics')
    print('=' * 60)
    for split, m in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
        print(f'{split:5s}  loss={m["loss"]:.4f}  acc={m["accuracy"]:.4f}  '
              f'mse={m["mse"]:.4f}  r2={m["r2"]:.4f}')
 
    save_artifacts(model, history, {'train': train_metrics,
                                    'val': val_metrics, 'test': test_metrics})
 
    # ── Quality assertions ────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Quality Assertions')
    print('=' * 60)
    all_passed = True
 
    # 1. Validation accuracy threshold
    if val_metrics['accuracy'] >= 0.87:
        print(f'PASS  val_accuracy={val_metrics["accuracy"]:.4f} >= 0.87')
    else:
        print(f'FAIL  val_accuracy={val_metrics["accuracy"]:.4f} < 0.87')
        all_passed = False
 
    # 2. LR warmup: lr should rise during first `WARMUP_EPOCHS` epochs
    warmup_lrs = history['lr'][:WARMUP_EPOCHS]
    if all(warmup_lrs[i] < warmup_lrs[i + 1] for i in range(len(warmup_lrs) - 1)):
        print('PASS  LR increases monotonically during warmup')
    else:
        print('FAIL  LR warmup not monotonically increasing')
        all_passed = False
 
    # 3. LR cosine: after warmup lr should decrease overall (first > last)
    post_warmup = history['lr'][WARMUP_EPOCHS:]
    if post_warmup[0] > post_warmup[-1]:
        print('PASS  LR decreases overall after warmup (cosine)')
    else:
        print('FAIL  LR did not decrease after warmup')
        all_passed = False
 
    # 4. Val loss decreased: first 5 avg > last 5 avg
    early_val = np.mean(history['val_loss'][:5])
    late_val  = np.mean(history['val_loss'][-5:])
    if late_val < early_val:
        print(f'PASS  Val loss decreased  ({early_val:.4f} -> {late_val:.4f})')
    else:
        print(f'FAIL  Val loss did not decrease ({early_val:.4f} -> {late_val:.4f})')
        all_passed = False
 
    print('\n' + '=' * 60)
    print(f'All checks passed: {all_passed}')
    print('=' * 60)
    sys.exit(0 if all_passed else 1)
