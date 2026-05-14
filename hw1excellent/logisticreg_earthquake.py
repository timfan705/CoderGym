"""
Logistic Regression — Earthquake Severity Classification
=========================================================
BigQuery dataset : bigquery-public-data.noaa_significant_earthquakes.earthquakes
Task             : Binary classification — predict whether an earthquake is
                   SIGNIFICANT (magnitude >= 6.0) based on depth, location,
                   and related seismic features.

Math
----
Sigmoid activation:
    sigma(z) = 1 / (1 + exp(-z))

Binary Cross-Entropy (Log-Loss):
    L(y, y_hat) = -[y * log(sigma(z)) + (1-y) * log(1 - sigma(z))]

pytorch_task_v1 protocol
------------------------
Entrypoint : python tasks/logreg_bigquery_earthquake/task.py
Required   : get_task_metadata, set_seed, get_device, make_dataloaders,
             build_model, train, evaluate, predict, save_artifacts
Exit       : sys.exit(0) on pass (val Accuracy >= 0.80, val R2 >= 0.20)
             sys.exit(1) on failure
"""

import sys
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report, r2_score
)

# ─────────────────────────────────────────────────────────────────────────────
# BigQuery SQL
# ─────────────────────────────────────────────────────────────────────────────
BIGQUERY_SQL = """
SELECT
    -- Target: 1 = significant (mag >= 6.0), 0 = minor
    CAST(magnitude >= 6.0 AS INT64)        AS is_significant,

    -- Seismic features
    magnitude,
    depth,
    COALESCE(latitude,  0.0)               AS latitude,
    COALESCE(longitude, 0.0)               AS longitude,

    -- Engineered: absolute latitude (distance from equator)
    ABS(COALESCE(latitude, 0.0))           AS abs_latitude,

    -- Tectonic proxy: depth category
    CASE
        WHEN depth < 70   THEN 0   -- shallow
        WHEN depth < 300  THEN 1   -- intermediate
        ELSE                   2   -- deep
    END                                    AS depth_category,

    -- Hemisphere flags
    CAST(COALESCE(latitude,  0) >= 0 AS INT64) AS northern_hemisphere,
    CAST(COALESCE(longitude, 0) >= 0 AS INT64) AS eastern_hemisphere

FROM `bigquery-public-data.noaa_significant_earthquakes.earthquakes`
WHERE
    magnitude IS NOT NULL
    AND depth   IS NOT NULL
    AND depth   >= 0
    AND magnitude BETWEEN 4.0 AND 10.0
"""

FEATURE_COLS = [
    "depth", "latitude", "longitude",
    "abs_latitude", "depth_category",
    "northern_hemisphere", "eastern_hemisphere",
]
TARGET_COL = "is_significant"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metadata
# ─────────────────────────────────────────────────────────────────────────────
def get_task_metadata() -> dict:
    return {
        "task_id":   "logreg_bigquery_earthquake",
        "series":    "Logistic Regression",
        "level":     2,
        "algorithm": "Logistic Regression (Binary, nn.Module)",
        "dataset": {
            "name":    "NOAA Significant Earthquakes",
            "source":  "bigquery-public-data.noaa_significant_earthquakes.earthquakes",
            "fallback":"synthetic (mirrors BigQuery schema)",
            "target":  "is_significant (mag >= 6.0)",
            "features": FEATURE_COLS,
        },
        "model": {
            "type":      "LogisticRegression",
            "loss":      "BCEWithLogitsLoss (pos_weight balanced)",
            "optimizer": "Adam",
            "lr":        1e-3,
            "epochs":    80,
            "batch_size": 64,
        },
        "pass_thresholds": {"val_accuracy": 0.80, "val_r2": 0.20},
        "exit_code": {"pass": 0, "fail": 1},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Device
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_bigquery() -> "pd.DataFrame":
    """Query BigQuery and return a DataFrame."""
    from google.cloud import bigquery
    import pandas as pd
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        raise EnvironmentError("Set GOOGLE_CLOUD_PROJECT to use BigQuery.")
    client = bigquery.Client(project=project)
    print("[BigQuery] Querying NOAA Significant Earthquakes …")
    df = client.query(BIGQUERY_SQL).to_dataframe()
    print(f"[BigQuery] Loaded {len(df):,} rows")
    return df


def _make_synthetic(n: int = 4_000, seed: int = 42) -> "pd.DataFrame":
    """
    Synthetic fallback that mirrors the BigQuery schema exactly.
    Significant earthquakes (mag >= 6) are the minority class (~25 %).
    """
    import pandas as pd
    rng = np.random.default_rng(seed)

    depth     = rng.exponential(scale=60, size=n).clip(1, 700)
    latitude  = rng.uniform(-90,  90, n)
    longitude = rng.uniform(-180, 180, n)

    # Significant quakes are deeper on average and cluster near ±30° lat
    log_odds = (
        -1.5
        + 0.003 * depth
        - 0.005 * np.abs(latitude - 30)
        + rng.normal(0, 0.4, n)
    )
    prob           = 1 / (1 + np.exp(-log_odds))
    is_significant = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "is_significant":     is_significant,
        "depth":              depth,
        "latitude":           latitude,
        "longitude":          longitude,
        "abs_latitude":       np.abs(latitude),
        "depth_category":     np.where(depth < 70, 0, np.where(depth < 300, 1, 2)),
        "northern_hemisphere":(latitude >= 0).astype(int),
        "eastern_hemisphere": (longitude >= 0).astype(int),
    })


def make_dataloaders(cfg: dict):
    """
    Returns (train_loader, val_loader, scaler, pos_weight_tensor, feature_names).
    Tries BigQuery first; falls back to synthetic data if unavailable.
    """
    import pandas as pd

    try:
        df = _load_bigquery()
    except Exception as exc:
        print(f"[INFO] BigQuery unavailable ({exc}). Using synthetic fallback.")
        df = _make_synthetic()

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    print(f"[Data] Samples={len(X):,}  Features={X.shape[1]}")
    print(f"       Class balance: {int(y.sum())} significant "
          f"/ {int((1-y).sum())} minor  "
          f"({y.mean()*100:.1f}% significant)")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg["val_frac"], stratify=y.astype(int),
        random_state=cfg["seed"])

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # Class-frequency weighting for imbalanced binary classification
    neg, pos   = (y_tr == 0).sum(), (y_tr == 1).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)

    def _loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.tensor(Xa), torch.tensor(ya))
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    return (_loader(X_tr, y_tr, True),
            _loader(X_val, y_val, False),
            scaler, pos_weight, FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model
# ─────────────────────────────────────────────────────────────────────────────
class LogisticRegressionModel(nn.Module):
    """
    Single linear layer for binary logistic regression.
    Outputs raw logit; sigmoid is applied by BCEWithLogitsLoss (train)
    and manually in predict().
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)   # shape [B]


def build_model(cfg: dict, n_features: int) -> nn.Module:
    return LogisticRegressionModel(n_features)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train (one epoch)
# ─────────────────────────────────────────────────────────────────────────────
def train(model: nn.Module,
          loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          device: torch.device) -> float:
    """Train for one epoch; returns mean BCE loss."""
    model.train()
    total_loss, total_n = 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logit = model(Xb)
        loss  = criterion(logit, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        total_n    += len(yb)
    return total_loss / total_n


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> dict:
    """
    Compute metrics on a DataLoader split.
    Returns dict with keys: loss, accuracy, roc_auc, f1, r2, mse,
                             confusion_matrix, classification_report.
    """
    model.eval()
    all_logits, all_labels = [], []
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logit = model(Xb)
            loss  = criterion(logit, yb)
            total_loss += loss.item() * len(yb)
            total_n    += len(yb)
            all_logits.append(logit.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(int)
    probs  = 1 / (1 + np.exp(-logits))          # sigmoid
    preds  = (probs >= 0.5).astype(int)

    # R2 and MSE are computed on probabilities vs binary labels
    mse = float(np.mean((probs - labels) ** 2))
    r2  = float(r2_score(labels, probs))

    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")

    return {
        "loss":                   total_loss / total_n,
        "accuracy":               float(accuracy_score(labels, preds)),
        "roc_auc":                auc,
        "f1_macro":               float(f1_score(labels, preds, average="macro",
                                                  zero_division=0)),
        "mse":                    mse,
        "r2":                     r2,
        "confusion_matrix":       confusion_matrix(labels, preds).tolist(),
        "classification_report":  classification_report(
                                      labels, preds,
                                      target_names=["Minor", "Significant"],
                                      zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Predict
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module,
            X: np.ndarray,
            device: torch.device,
            threshold: float = 0.5):
    """
    Returns (predictions [0/1], probabilities [0–1]) for raw numpy input.
    """
    model.eval()
    Xt    = torch.tensor(X, dtype=torch.float32).to(device)
    logit = model(Xt).cpu().numpy()
    probs = 1 / (1 + np.exp(-logit))
    preds = (probs >= threshold).astype(int)
    return preds, probs


# ─────────────────────────────────────────────────────────────────────────────
# 9. Save artifacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(model: nn.Module,
                   metrics: dict,
                   cfg: dict,
                   out_dir: str = ".") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Model weights
    torch.save(model.state_dict(),
               os.path.join(out_dir, "logreg_earthquake_weights.pt"))

    # Metrics JSON
    def _serial(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    payload = {"cfg": cfg,
               "train_metrics": metrics.get("train", {}),
               "val_metrics":   metrics.get("val", {})}
    with open(os.path.join(out_dir, "logreg_earthquake_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_serial)

    # Feature weights
    w = model.linear.weight.data.cpu().numpy().flatten()
    b = model.linear.bias.data.cpu().item()
    lines  = ["Feature Weights (logistic regression coefficients)\n",
               "=" * 52 + "\n"]
    ranked = sorted(zip(cfg.get("feature_names", []), w),
                    key=lambda t: abs(t[1]), reverse=True)
    for name, wi in ranked:
        bar = "█" * int(abs(wi) * 15)
        lines.append(f"  {name:<25} {wi:+.4f}  {bar}\n")
    lines.append(f"  {'bias':<25} {b:+.4f}\n")
    with open(os.path.join(out_dir, "logreg_earthquake_weights.txt"), "w") as f:
        f.writelines(lines)

    print(f"[Artifacts] Saved to '{out_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = {
        "seed":       42,
        "val_frac":   0.20,
        "batch_size": 64,
        "lr":         1e-3,
        "epochs":     80,
        "out_dir":    "artifacts/logreg_bigquery_earthquake",
        # quality gates
        "min_val_accuracy": 0.80,
        "min_val_r2":       0.20,
    }

    meta = get_task_metadata()
    print("=" * 60)
    print(f"Task : {meta['task_id']}")
    print(f"Goal : {meta['dataset']['target']}")
    print("=" * 60)

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(cfg["seed"])
    device = get_device()
    print(f"[Device] {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, scaler, pos_weight, feat_names = \
        make_dataloaders(cfg)
    cfg["feature_names"] = feat_names
    n_features = len(feat_names)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model(cfg, n_features).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-5)

    print(f"\n[Model] {model}")
    print(f"        Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  "
          f"{'Val Acc':>9}  {'Val AUC':>9}  {'Val F1':>8}")
    print("-" * 62)

    loss_history, val_loss_history = [], []
    best_val_acc, best_val_auc = 0.0, 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss  = train(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        loss_history.append(tr_loss)
        val_loss_history.append(val_metrics["loss"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
        if not math.isnan(val_metrics["roc_auc"]) and \
                val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>12.5f}  "
                  f"{val_metrics['loss']:>10.5f}  "
                  f"{val_metrics['accuracy']:>9.4f}  "
                  f"{val_metrics['roc_auc']:>9.4f}  "
                  f"{val_metrics['f1_macro']:>8.4f}")

    # ── Final evaluation on both splits ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics_f = evaluate(model, val_loader,   criterion, device)

    for split, m in [("Train", train_metrics), ("Val", val_metrics_f)]:
        print(f"\n[{split}]")
        print(f"  Loss        : {m['loss']:.5f}")
        print(f"  MSE         : {m['mse']:.5f}")
        print(f"  R²          : {m['r2']:.4f}")
        print(f"  Accuracy    : {m['accuracy']:.4f}")
        print(f"  ROC-AUC     : {m['roc_auc']:.4f}")
        print(f"  Macro-F1    : {m['f1_macro']:.4f}")

    print("\n[Val Classification Report]")
    print(val_metrics_f["classification_report"])
    print("[Val Confusion Matrix]  rows=true  cols=pred")
    print(np.array(val_metrics_f["confusion_matrix"]))

    # ── Feature weights ───────────────────────────────────────────────────────
    w      = model.linear.weight.data.cpu().numpy().flatten()
    b      = model.linear.bias.data.cpu().item()
    ranked = sorted(zip(feat_names, w), key=lambda t: abs(t[1]), reverse=True)
    print("\n[Learned Feature Weights  (|magnitude| ranked)]")
    for name, wi in ranked:
        bar = "█" * int(abs(wi) * 15)
        print(f"  {name:<25} {wi:+.6f}  {bar}")
    print(f"  {'bias':<25} {b:+.6f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    save_artifacts(model,
                   {"train": train_metrics, "val": val_metrics_f},
                   cfg,
                   out_dir=cfg["out_dir"])

    # ── Quality assertions (self-verifiable exit) ──────────────────────────────
    print("\n[Quality Gates]")
    val_acc = val_metrics_f["accuracy"]
    val_r2  = val_metrics_f["r2"]
    passed_acc = val_acc >= cfg["min_val_accuracy"]
    passed_r2  = val_r2  >= cfg["min_val_r2"]

    print(f"  Val Accuracy {val_acc:.4f} >= {cfg['min_val_accuracy']} "
          f"→ {'✓ PASS' if passed_acc else '✗ FAIL'}")
    print(f"  Val R²       {val_r2:.4f} >= {cfg['min_val_r2']} "
          f"→ {'✓ PASS' if passed_r2 else '✗ FAIL'}")
    print(f"  Best Val Accuracy  : {best_val_acc:.4f}")
    print(f"  Best Val ROC-AUC   : {best_val_auc:.4f}")

    assert passed_acc, \
        f"FAIL: val accuracy {val_acc:.4f} < threshold {cfg['min_val_accuracy']}"
    assert passed_r2, \
        f"FAIL: val R² {val_r2:.4f} < threshold {cfg['min_val_r2']}"

    print("\n[Exit] exit_code=0  ALL QUALITY GATES PASSED ✓")
    sys.exit(0)
