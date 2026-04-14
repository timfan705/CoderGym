import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
 
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
        'task_id':   'tfm_labelsmooth_agnews',
        'series':    'Transformers',
        'algorithm': 'Transformer Encoder + Label Smoothing',
        'dataset':   'AG News (torchtext)',
        'metrics':   ['accuracy', 'macro_f1', 'loss', 'mse', 'r2'],
        'threshold': {'val_accuracy': 0.88},
    }
 
# ── AG News dataset (torchtext) ───────────────────────────────────────────────
def build_vocab_and_dataset(max_vocab: int = 20_000, max_len: int = 64):
    """
    Load AG News via torchtext.  Falls back to a small synthetic corpus when
    torchtext is not available (rare offline environments).
    Returns (train_ds, test_ds, vocab_size, n_classes).
    """
    try:
        from torchtext.datasets import AG_NEWS
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
 
        tokenizer = get_tokenizer('basic_english')
 
        def _yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
 
        # Build vocab from training split
        train_iter = AG_NEWS(root='/tmp/agnews', split='train')
        vocab = build_vocab_from_iterator(
            _yield_tokens(train_iter),
            max_tokens=max_vocab,
            specials=['<pad>', '<unk>'])
        vocab.set_default_index(vocab['<unk>'])
 
        def encode(text):
            tokens = tokenizer(text)[:max_len]
            ids    = vocab(tokens)
            # Pad / truncate to max_len
            ids = ids + [0] * (max_len - len(ids))
            return ids[:max_len]
 
        class AGNewsDataset(Dataset):
            def __init__(self, split):
                self.samples = []
                for label, text in AG_NEWS(root='/tmp/agnews', split=split):
                    self.samples.append((encode(text), int(label) - 1))
 
            def __len__(self):  return len(self.samples)
            def __getitem__(self, idx):
                x, y = self.samples[idx]
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
 
        train_ds = AGNewsDataset('train')
        test_ds  = AGNewsDataset('test')
        return train_ds, test_ds, min(max_vocab, len(vocab)), 4
 
    except Exception:
        # ── Synthetic fallback: 4 "topics", bigram-style vocab ────────────────
        print('[WARN] torchtext/AG_NEWS unavailable – using synthetic fallback')
        rng = np.random.default_rng(42)
        n_classes, vocab_size = 4, 1000
        n_train, n_test = 5000, 1000
 
        def _synth_split(n):
            labels = rng.integers(0, n_classes, size=n)
            texts  = []
            for lbl in labels:
                # Class-specific token cluster + noise
                cluster = (lbl * 200 + rng.integers(0, 200, size=30)).tolist()
                noise   = rng.integers(0, vocab_size, size=34).tolist()
                ids     = (cluster + noise)[:max_len]
                ids    += [0] * (max_len - len(ids))
                texts.append(ids)
            return list(zip(texts, labels.tolist()))
 
        class SynthDS(Dataset):
            def __init__(self, pairs):
                self.pairs = pairs
            def __len__(self): return len(self.pairs)
            def __getitem__(self, idx):
                x, y = self.pairs[idx]
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
 
        return SynthDS(_synth_split(n_train)), SynthDS(_synth_split(n_test)), vocab_size, n_classes
 
def make_dataloaders(batch_size: int = 128, val_fraction: float = 0.05):
    train_ds, test_ds, vocab_size, n_classes = build_vocab_and_dataset()
 
    n_val   = max(1, int(len(train_ds) * val_fraction))
    n_train = len(train_ds) - n_val
    train_sub, val_sub = random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))
 
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, vocab_size, n_classes
 
# ── Label-Smoothing Loss ──────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing  = smoothing
        self.n_classes  = n_classes
        self.confidence = 1.0 - smoothing
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        # Hard targets
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Uniform smoothing
        smooth = -log_probs.mean(dim=-1)
        loss = self.confidence * nll + self.smoothing * smooth
        return loss.mean()
 
# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])
 
# ── Transformer Encoder Classifier ───────────────────────────────────────────
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, n_classes: int,
                 d_model: int = 128, nhead: int = 4,
                 n_layers: int = 2, dim_ff: int = 256,
                 max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos   = PositionalEncoding(d_model, dropout, max_len)
        enc_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len)
        pad_mask = (x == 0)                           # True = ignore
        e = self.pos(self.embed(x))                   # (B, seq, d_model)
        h = self.encoder(e, src_key_padding_mask=pad_mask)
        # CLS: mean pool over non-padding tokens
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.classifier(pooled)
 
def build_model(vocab_size: int, n_classes: int, device: torch.device) -> nn.Module:
    return TransformerClassifier(vocab_size, n_classes).to(device)
 
# ── Train / Evaluate ──────────────────────────────────────────────────────────
def train(model: nn.Module, loader: DataLoader, optimizer,
          criterion, device: torch.device) -> float:
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)
 
def evaluate(model: nn.Module, loader: DataLoader,
             criterion, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    n_classes = None
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            if n_classes is None:
                n_classes = logits.shape[1]
            total_loss += criterion(logits, y).item() * len(X)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_targets.append(y.cpu().numpy())
 
    targets = np.concatenate(all_targets)
    preds   = np.concatenate(all_preds)
    probs   = np.concatenate(all_probs)
    oh      = np.zeros((len(targets), n_classes))
    oh[np.arange(len(targets)), targets] = 1.0
 
    return {
        'loss':     total_loss / len(loader.dataset),
        'accuracy': float(accuracy_score(targets, preds)),
        'macro_f1': float(f1_score(targets, preds, average='macro', zero_division=0)),
        'mse':      float(mean_squared_error(oh, probs)),
        'r2':       float(r2_score(oh, probs)),
    }
 
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            preds.append(model(X.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)
 
# ── Artifacts ─────────────────────────────────────────────────────────────────
def save_artifacts(model, history_ls, history_base, metrics_ls, metrics_base):
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'tfm_agnews.pt'))
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'label_smooth': metrics_ls, 'baseline': metrics_base}, f, indent=2)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, h, c in [('Label Smooth', history_ls, 'blue'),
                         ('No Smooth',   history_base, 'red')]:
        axes[0].plot(h['val_loss'], label=label, color=c)
        axes[1].plot(h['val_acc'],  label=label, color=c)
    for ax, title in zip(axes, ['Val Loss', 'Val Accuracy']):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tfm_labelsmooth_agnews_curves.png'), dpi=100)
    plt.close()
    print(f'Artifacts saved to {OUTPUT_DIR}')
 
# ── Training helper ───────────────────────────────────────────────────────────
def run_training(use_label_smooth: bool, epochs: int, device,
                 train_loader, val_loader, vocab_size, n_classes,
                 smoothing: float = 0.1, lr: float = 3e-4):
    set_seed(42)
    model     = build_model(vocab_size, n_classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = (LabelSmoothingCE(n_classes, smoothing)
                 if use_label_smooth else nn.CrossEntropyLoss())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=len(train_loader))
    history   = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    tag = 'LabelSmooth' if use_label_smooth else 'Baseline'
 
    for epoch in range(1, epochs + 1):
        t_loss = train(model, train_loader, optimizer, criterion, device)
        scheduler.step()   # called once per epoch here for simplicity
        v_met  = evaluate(model, val_loader, criterion, device)
        t_met  = evaluate(model, train_loader, criterion, device)
        history['train_loss'].append(t_met['loss'])
        history['val_loss'].append(v_met['loss'])
        history['train_acc'].append(t_met['accuracy'])
        history['val_acc'].append(v_met['accuracy'])
        if epoch % 3 == 0 or epoch == 1:
            print(f'[{tag}] Epoch {epoch:2d}/{epochs}  '
                  f'train_loss={t_met["loss"]:.4f}  val_loss={v_met["loss"]:.4f}  '
                  f'val_acc={v_met["accuracy"]:.4f}  macro_f1={v_met["macro_f1"]:.4f}')
    return model, history
 
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    EPOCHS = 15
    print('=' * 60)
    print('Transformer Encoder + Label Smoothing  |  AG News')
    print('=' * 60)
    print(f'Device: {device}')
 
    train_loader, val_loader, test_loader, vocab_size, n_classes = make_dataloaders(128)
    print(f'Vocab size: {vocab_size}  |  Classes: {n_classes}')
    print(f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}')
 
    # ── With label smoothing ───────────────────────────────────────────────────
    print('\n--- Training WITH Label Smoothing ---')
    ls_model, ls_hist = run_training(
        use_label_smooth=True, epochs=EPOCHS, device=device,
        train_loader=train_loader, val_loader=val_loader,
        vocab_size=vocab_size, n_classes=n_classes)
 
    # ── Without label smoothing (baseline) ────────────────────────────────────
    print('\n--- Training WITHOUT Label Smoothing (baseline) ---')
    base_model, base_hist = run_training(
        use_label_smooth=False, epochs=EPOCHS, device=device,
        train_loader=train_loader, val_loader=val_loader,
        vocab_size=vocab_size, n_classes=n_classes)
 
    # ── Final evaluation ──────────────────────────────────────────────────────
    hard_crit = nn.CrossEntropyLoss()
    ls_val   = evaluate(ls_model,   val_loader,  hard_crit, device)
    base_val = evaluate(base_model, val_loader,  hard_crit, device)
    ls_test  = evaluate(ls_model,   test_loader, hard_crit, device)
 
    print('\n' + '=' * 60)
    print('Final Metrics (Validation)')
    print('=' * 60)
    for tag, m in [('LabelSmooth', ls_val), ('Baseline', base_val)]:
        print(f'{tag:12s}  acc={m["accuracy"]:.4f}  macro_f1={m["macro_f1"]:.4f}  '
              f'loss={m["loss"]:.4f}  mse={m["mse"]:.4f}  r2={m["r2"]:.4f}')
 
    save_artifacts(ls_model, ls_hist, base_hist,
                   {'val': ls_val, 'test': ls_test}, {'val': base_val})
 
    # ── Quality assertions ────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Quality Assertions')
    print('=' * 60)
    all_passed = True
 
    if ls_val['accuracy'] >= 0.88:
        print(f'PASS  LabelSmooth val_acc={ls_val["accuracy"]:.4f} >= 0.88')
    else:
        print(f'FAIL  LabelSmooth val_acc={ls_val["accuracy"]:.4f} < 0.88')
        all_passed = False
 
    gap = ls_val['accuracy'] - base_val['accuracy']
    if gap >= -0.005:
        print(f'PASS  Label-smooth acc gap vs baseline = {gap:+.4f} (>= -0.005)')
    else:
        print(f'FAIL  Label-smooth acc gap vs baseline = {gap:+.4f} (< -0.005)')
        all_passed = False
 
    if ls_val['macro_f1'] >= 0.85:
        print(f'PASS  macro_f1={ls_val["macro_f1"]:.4f} >= 0.85')
    else:
        print(f'FAIL  macro_f1={ls_val["macro_f1"]:.4f} < 0.85')
        all_passed = False
 
    early = np.mean(ls_hist['val_loss'][:3])
    late  = np.mean(ls_hist['val_loss'][-3:])
    if late < early:
        print(f'PASS  Val loss decreased ({early:.4f} -> {late:.4f})')
    else:
        print(f'FAIL  Val loss did not decrease ({early:.4f} -> {late:.4f})')
        all_passed = False
 
    print('\n' + '=' * 60)
    print(f'All checks passed: {all_passed}')
    print('=' * 60)
    sys.exit(0 if all_passed else 1)
