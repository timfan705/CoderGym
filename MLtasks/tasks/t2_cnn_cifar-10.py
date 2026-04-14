import os
import sys
import json
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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
 
# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def get_task_metadata() -> dict:
    return {
        'task_id': 'cnn_mixup_cifar10',
        'series': 'Convolutional Neural Networks',
        'algorithm': 'CNN with Mixup Data Augmentation',
        'dataset': 'CIFAR-10',
        'metrics': ['accuracy', 'loss', 'mse', 'r2'],
        'threshold': {'val_accuracy': 0.80},
    }
 
# ── Data ──────────────────────────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
 
def make_dataloaders(batch_size: int = 128, val_fraction: float = 0.1):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
 
    full_train = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=True, download=True, transform=train_tf)
    test_set   = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=False, download=True, transform=test_tf)
 
    n_val   = int(len(full_train) * val_fraction)
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
 
# ── Model: simple ResNet-like CNN ─────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
 
    def forward(self, x):
        return F.relu(x + self.block(x))
 
class SmallResNet(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock(64),  nn.MaxPool2d(2))   # 16x16
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128), nn.MaxPool2d(2))                             # 8x8
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ResBlock(256), nn.AdaptiveAvgPool2d(1))                     # 1x1
        self.classifier = nn.Linear(256, n_classes)
 
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x.view(x.size(0), -1))
 
def build_model(device: torch.device) -> nn.Module:
    return SmallResNet().to(device)
 
# ── Mixup ─────────────────────────────────────────────────────────────────────
def mixup_batch(X: torch.Tensor, y: torch.Tensor,
                alpha: float = 0.4, n_classes: int = 10):
    """Return mixed (X, y_a, y_b, lam) for a batch."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(X.size(0), device=X.device)
    X_mix = lam * X + (1 - lam) * X[idx]
    return X_mix, y, y[idx], lam
 
def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
 
# ── Train / Evaluate ──────────────────────────────────────────────────────────
def train(model: nn.Module, loader: DataLoader, optimizer, criterion,
          device: torch.device, use_mixup: bool = True, alpha: float = 0.4) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if use_mixup:
            X_mix, y_a, y_b, lam = mixup_batch(X, y, alpha)
            logits = model(X_mix)
            loss   = mixup_loss(criterion, logits, y_a, y_b, lam)
        else:
            loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)
 
def evaluate(model: nn.Module, loader: DataLoader, criterion,
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
    n_cls   = probs.shape[1]
    oh = np.zeros((len(targets), n_cls)); oh[np.arange(len(targets)), targets] = 1.0
 
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
def save_artifacts(model, history_mix, history_base, metrics_mix, metrics_base):
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'cnn_mixup_cifar10.pt'))
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'mixup': metrics_mix, 'baseline': metrics_base}, f, indent=2)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, h, c in [('Mixup', history_mix, 'blue'), ('Baseline', history_base, 'red')]:
        axes[0].plot(h['val_loss'], label=label, color=c)
        axes[1].plot(h['val_acc'],  label=label, color=c)
    for ax, title in zip(axes, ['Val Loss', 'Val Accuracy']):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_mixup_cifar10_curves.png'), dpi=100)
    plt.close()
    print(f'Artifacts saved to {OUTPUT_DIR}')
 
# ── Training loop helper ──────────────────────────────────────────────────────
def run_training(use_mixup: bool, epochs: int, device, train_loader,
                 val_loader, lr: float = 1e-3):
    set_seed(42)
    model     = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader))
    history   = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    tag = 'Mixup' if use_mixup else 'Baseline'
 
    for epoch in range(1, epochs + 1):
        t_loss = train(model, train_loader, optimizer, criterion,
                       device, use_mixup=use_mixup)
        # OneCycleLR steps per batch; call once more at epoch end for logging
        v_met  = evaluate(model, val_loader, criterion, device)
        t_met  = evaluate(model, train_loader, criterion, device)
        history['train_loss'].append(t_met['loss'])
        history['val_loss'].append(v_met['loss'])
        history['train_acc'].append(t_met['accuracy'])
        history['val_acc'].append(v_met['accuracy'])
        if epoch % 5 == 0 or epoch == 1:
            print(f'[{tag}] Epoch {epoch:3d}/{epochs}  '
                  f'train_loss={t_met["loss"]:.4f}  val_loss={v_met["loss"]:.4f}  '
                  f'val_acc={v_met["accuracy"]:.4f}')
    return model, history
 
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    EPOCHS = 25
    print('=' * 60)
    print('CNN + Mixup Augmentation  |  CIFAR-10')
    print('=' * 60)
    print(f'Device: {device}')
 
    train_loader, val_loader, test_loader = make_dataloaders(128)
 
    # ── Train Mixup model ─────────────────────────────────────────────────────
    print('\n--- Training WITH Mixup ---')
    mix_model, mix_hist = run_training(
        use_mixup=True, epochs=EPOCHS, device=device,
        train_loader=train_loader, val_loader=val_loader)
 
    # ── Train Baseline model ──────────────────────────────────────────────────
    print('\n--- Training WITHOUT Mixup (baseline) ---')
    base_model, base_hist = run_training(
        use_mixup=False, epochs=EPOCHS, device=device,
        train_loader=train_loader, val_loader=val_loader)
 
    # ── Final evaluation ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    mix_val   = evaluate(mix_model,  val_loader,  criterion, device)
    base_val  = evaluate(base_model, val_loader,  criterion, device)
    mix_test  = evaluate(mix_model,  test_loader, criterion, device)
 
    print('\n' + '=' * 60)
    print('Final Metrics (Validation)')
    print('=' * 60)
    for tag, m in [('Mixup', mix_val), ('Baseline', base_val)]:
        print(f'{tag:8s}  acc={m["accuracy"]:.4f}  loss={m["loss"]:.4f}  '
              f'mse={m["mse"]:.4f}  r2={m["r2"]:.4f}')
 
    save_artifacts(mix_model, mix_hist, base_hist,
                   {'val': mix_val, 'test': mix_test}, {'val': base_val})
 
    # ── Quality assertions ────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Quality Assertions')
    print('=' * 60)
    all_passed = True
 
    if mix_val['accuracy'] >= 0.80:
        print(f'PASS  Mixup val_acc={mix_val["accuracy"]:.4f} >= 0.80')
    else:
        print(f'FAIL  Mixup val_acc={mix_val["accuracy"]:.4f} < 0.80')
        all_passed = False
 
    # Mixup should not hurt accuracy by more than 1 pp vs baseline
    gap = mix_val['accuracy'] - base_val['accuracy']
    if gap >= -0.01:
        print(f'PASS  Mixup acc gap vs baseline = {gap:+.4f} (>= -0.01)')
    else:
        print(f'FAIL  Mixup acc gap vs baseline = {gap:+.4f} (< -0.01)')
        all_passed = False
 
    # Val loss should decrease
    early = np.mean(mix_hist['val_loss'][:3])
    late  = np.mean(mix_hist['val_loss'][-3:])
    if late < early:
        print(f'PASS  Mixup val loss decreased ({early:.4f} -> {late:.4f})')
    else:
        print(f'FAIL  Mixup val loss did not decrease ({early:.4f} -> {late:.4f})')
        all_passed = False
 
    print('\n' + '=' * 60)
    print(f'All checks passed: {all_passed}')
    print('=' * 60)
    sys.exit(0 if all_passed else 1)
