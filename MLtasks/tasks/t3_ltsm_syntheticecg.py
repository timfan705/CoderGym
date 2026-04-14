import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
 
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
        'task_id':   'lstm_timeseries_ecg',
        'series':    'Sequence Models (RNN/LSTM)',
        'algorithm': 'LSTM Multi-Step Forecasting',
        'dataset':   'Synthetic ECG-like signal',
        'metrics':   ['mse', 'r2', 'loss'],
        'threshold': {'val_mse': 0.05, 'val_r2': 0.80},
    }
 
# ── Synthetic ECG-like data ───────────────────────────────────────────────────
def generate_ecg_signal(n_samples: int = 8000, fs: int = 100,
                         noise_std: float = 0.05, seed: int = 42) -> np.ndarray:
    """
    Simulate a simplified ECG: sum of a dominant sine (heart beat ~1.2 Hz),
    a narrower QRS-like spike train, and mild noise.
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n_samples) / fs
    # Slow sinusoidal base (P + T waves analogue)
    base = 0.5 * np.sin(2 * np.pi * 1.2 * t)
    # QRS-like narrow spikes every ~0.83 s
    spikes = np.zeros(n_samples)
    period = int(fs / 1.2)
    for i in range(0, n_samples, period):
        w = 3  # half-width samples
        for d in range(-w, w + 1):
            if 0 <= i + d < n_samples:
                spikes[i + d] += np.exp(-0.5 * (d / (w / 3)) ** 2)
    signal = base + spikes + rng.normal(0, noise_std, n_samples)
    # Normalise to [-1, 1]
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    return signal.astype(np.float32)
 
class TimeSeriesDataset(Dataset):
    def __init__(self, signal: np.ndarray, seq_len: int = 100, horizon: int = 20):
        self.X, self.y = [], []
        for i in range(len(signal) - seq_len - horizon + 1):
            self.X.append(signal[i: i + seq_len])
            self.y.append(signal[i + seq_len: i + seq_len + horizon])
        self.X = torch.tensor(np.array(self.X)).unsqueeze(-1)  # (N, seq, 1)
        self.y = torch.tensor(np.array(self.y))                # (N, horizon)
 
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
 
def make_dataloaders(batch_size: int = 64, seq_len: int = 100, horizon: int = 20):
    signal = generate_ecg_signal()
    n      = len(signal)
    train_sig = signal[:int(0.7 * n)]
    val_sig   = signal[int(0.7 * n): int(0.85 * n)]
    test_sig  = signal[int(0.85 * n):]
 
    train_ds = TimeSeriesDataset(train_sig, seq_len, horizon)
    val_ds   = TimeSeriesDataset(val_sig,   seq_len, horizon)
    test_ds  = TimeSeriesDataset(test_sig,  seq_len, horizon)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, horizon
 
# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """Stacked LSTM → FC head for multi-step forecasting."""
 
    def __init__(self, input_size: int = 1, hidden: int = 128,
                 n_layers: int = 2, horizon: int = 20, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (B, seq, hidden)
        return self.head(out[:, -1, :]) # (B, horizon)
 
def build_model(horizon: int, device: torch.device) -> nn.Module:
    return LSTMForecaster(horizon=horizon).to(device)
 
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
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += criterion(pred, y).item() * len(X)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
 
    preds   = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    mse     = float(mean_squared_error(targets, preds))
    r2      = float(r2_score(targets, preds))
    return {'loss': total_loss / len(loader.dataset), 'mse': mse, 'r2': r2}
 
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            preds.append(model(X.to(device)).cpu().numpy())
    return np.concatenate(preds)
 
# ── Naive persistence baseline ────────────────────────────────────────────────
def persistence_mse(loader: DataLoader) -> float:
    """Predict each future step == last observed value (naive baseline)."""
    errs = []
    for X, y in loader:
        last = X[:, -1, 0].numpy()                    # (B,)
        pred = np.tile(last[:, None], (1, y.shape[1])) # (B, horizon)
        errs.append(mean_squared_error(y.numpy().flatten(), pred.flatten()))
    return float(np.mean(errs))
 
# ── Artifacts ─────────────────────────────────────────────────────────────────
def save_artifacts(model, history, val_metrics, signal, loader, device):
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'lstm_ecg.pt'))
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(val_metrics, f, indent=2)
 
    # -- Forecast sample plot
    model.eval()
    sample_X, sample_y = next(iter(loader))
    with torch.no_grad():
        sample_pred = model(sample_X[:1].to(device)).cpu().numpy()[0]
    context = sample_X[0, :, 0].numpy()
    target  = sample_y[0].numpy()
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    t_ctx = np.arange(len(context))
    t_fut = np.arange(len(context), len(context) + len(target))
    axes[0].plot(t_ctx, context, label='Context')
    axes[0].plot(t_fut, target,      label='True Future', color='green')
    axes[0].plot(t_fut, sample_pred, label='Predicted',   color='red', linestyle='--')
    axes[0].set_title('ECG Forecast Sample')
    axes[0].legend()
 
    axes[1].plot(history['train_loss'], label='Train')
    axes[1].plot(history['val_loss'],   label='Val')
    axes[1].set_title('Loss Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
 
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lstm_ecg_results.png'), dpi=100)
    plt.close()
    print(f'Artifacts saved to {OUTPUT_DIR}')
 
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    EPOCHS     = 40
    BATCH_SIZE = 64
    SEQ_LEN    = 100
    HORIZON    = 20
 
    print('=' * 60)
    print('LSTM Multi-Step Forecasting  |  Synthetic ECG')
    print('=' * 60)
    print(f'Device: {device}')
 
    train_loader, val_loader, test_loader, horizon = make_dataloaders(
        BATCH_SIZE, SEQ_LEN, HORIZON)
 
    model     = build_model(horizon, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False)
 
    history = dict(train_loss=[], val_loss=[], val_mse=[], val_r2=[])
 
    for epoch in range(1, EPOCHS + 1):
        t_loss = train(model, train_loader, optimizer, criterion, device)
        v_met  = evaluate(model, val_loader, criterion, device)
        scheduler.step(v_met['loss'])
 
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_met['loss'])
        history['val_mse'].append(v_met['mse'])
        history['val_r2'].append(v_met['r2'])
 
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:3d}/{EPOCHS}  '
                  f'train_loss={t_loss:.5f}  val_mse={v_met["mse"]:.5f}  '
                  f'val_r2={v_met["r2"]:.4f}')
 
    # ── Final evaluation ──────────────────────────────────────────────────────
    train_met = evaluate(model, train_loader, criterion, device)
    val_met   = evaluate(model, val_loader,   criterion, device)
    test_met  = evaluate(model, test_loader,  criterion, device)
    pers_mse  = persistence_mse(val_loader)
 
    print('\n' + '=' * 60)
    print('Final Metrics')
    print('=' * 60)
    for split, m in [('Train', train_met), ('Val', val_met), ('Test', test_met)]:
        print(f'{split:5s}  mse={m["mse"]:.5f}  r2={m["r2"]:.4f}  loss={m["loss"]:.5f}')
    print(f'Persistence baseline MSE: {pers_mse:.5f}')
 
    signal = generate_ecg_signal()
    save_artifacts(model, history,
                   {'train': train_met, 'val': val_met, 'test': test_met,
                    'persistence_mse': pers_mse},
                   signal, val_loader, device)
 
    # ── Quality assertions ────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Quality Assertions')
    print('=' * 60)
    all_passed = True
 
    if val_met['mse'] < 0.05:
        print(f'PASS  val_mse={val_met["mse"]:.5f} < 0.05')
    else:
        print(f'FAIL  val_mse={val_met["mse"]:.5f} >= 0.05')
        all_passed = False
 
    if val_met['r2'] > 0.80:
        print(f'PASS  val_r2={val_met["r2"]:.4f} > 0.80')
    else:
        print(f'FAIL  val_r2={val_met["r2"]:.4f} <= 0.80')
        all_passed = False
 
    if val_met['mse'] < pers_mse:
        print(f'PASS  LSTM MSE ({val_met["mse"]:.5f}) < Persistence MSE ({pers_mse:.5f})')
    else:
        print(f'FAIL  LSTM MSE ({val_met["mse"]:.5f}) >= Persistence MSE ({pers_mse:.5f})')
        all_passed = False
 
    # Loss decreased overall
    early = np.mean(history['val_loss'][:5])
    late  = np.mean(history['val_loss'][-5:])
    if late < early:
        print(f'PASS  Val loss decreased ({early:.5f} -> {late:.5f})')
    else:
        print(f'FAIL  Val loss did not decrease ({early:.5f} -> {late:.5f})')
        all_passed = False
 
    print('\n' + '=' * 60)
    print(f'All checks passed: {all_passed}')
    print('=' * 60)
    sys.exit(0 if all_passed else 1)
