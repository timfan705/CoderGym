"""
Linear Regression — Housing Price Prediction
=============================================
BigQuery dataset : bigquery-public-data.ml_datasets.ames_housing
Task             : Regression — predict SalePrice from structural and
                   neighbourhood features of residential properties in
                   Ames, Iowa.

Math
----
Hypothesis (multivariate linear regression):
    h_theta(x) = theta_0 + theta_1*x_1 + theta_2*x_2 + ... + theta_n*x_n

MSE Cost Function:
    J(theta) = (1 / 2m) * sum_{i=1}^{m} (h_theta(x^i) - y^i)^2

Gradient w.r.t. theta_j (used by autograd):
    dJ/d(theta_j) = (1/m) * sum_{i=1}^{m} (h_theta(x^i) - y^i) * x_j^i

pytorch_task_v1 protocol
------------------------
Entrypoint : python tasks/linreg_bigquery_housing/task.py
Required   : get_task_metadata, set_seed, get_device, make_dataloaders,
             build_model, train, evaluate, predict, save_artifacts
Exit       : sys.exit(0) on pass (val R2 >= 0.70, val MSE printed)
             sys.exit(1) on failure
"""

import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# BigQuery SQL  (Ames Housing from ml_datasets public dataset)
# ─────────────────────────────────────────────────────────────────────────────
BIGQUERY_SQL = """
SELECT
    -- Target (log-transform applied in Python for normality)
    sale_price,

    -- Size features
    gr_liv_area          AS living_area_sqft,
    total_bsmt_sf        AS basement_sqft,
    COALESCE(garage_area, 0)       AS garage_sqft,
    lot_area,

    -- Quality / condition (ordinal 1-10)
    overall_qual,
    overall_cond,

    -- Age features
    year_built,
    year_remod_add       AS year_remodelled,
    (2010 - year_built)  AS house_age_years,

    -- Room counts
    full_bath,
    bedroom_abv_gr       AS bedrooms,
    tot_rms_abv_grd      AS total_rooms,

    -- Garage
    COALESCE(garage_cars, 0)       AS garage_capacity,

    -- Fireplace proxy
    fireplaces

FROM `bigquery-public-data.ml_datasets.ames_housing`
WHERE
    sale_price    IS NOT NULL
    AND sale_price > 0
    AND gr_liv_area IS NOT NULL
    AND overall_qual IS NOT NULL
"""

FEATURE_COLS = [
    "living_area_sqft", "basement_sqft",  "garage_sqft",
    "lot_area",
    "overall_qual",     "overall_cond",
    "house_age_years",
    "full_bath",        "bedrooms",        "total_rooms",
    "garage_capacity",  "fireplaces",
]
TARGET_COL = "sale_price"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metadata
# ─────────────────────────────────────────────────────────────────────────────
def get_task_metadata() -> dict:
    return {
        "task_id":   "linreg_bigquery_housing",
        "series":    "Linear Regression",
        "level":     4,
        "algorithm": "Linear Regression (Multivariate, nn.Module)",
        "dataset": {
            "name":     "Ames Housing",
            "source":   "bigquery-public-data.ml_datasets.ames_housing",
            "fallback": "synthetic (mirrors Ames Housing schema)",
            "target":   "log(SalePrice)",
            "features": FEATURE_COLS,
        },
        "model": {
            "type":      "LinearRegression",
            "loss":      "MSELoss (on log-price)",
            "optimizer": "Adam + weight_decay (L2)",
            "lr":        5e-3,
            "epochs":    120,
            "batch_size": 64,
        },
        "pass_thresholds": {"val_r2": 0.70, "val_mse_log": 0.10},
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
# 4. Data loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_bigquery() -> "pd.DataFrame":
    from google.cloud import bigquery
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        raise EnvironmentError("Set GOOGLE_CLOUD_PROJECT to use BigQuery.")
    client = bigquery.Client(project=project)
    print("[BigQuery] Querying Ames Housing dataset …")
    df = client.query(BIGQUERY_SQL).to_dataframe()
    print(f"[BigQuery] Loaded {len(df):,} rows")
    return df


def _make_synthetic(n: int = 2_000, seed: int = 42) -> "pd.DataFrame":
    """
    Synthetic housing data that mirrors the Ames Housing schema.
    Relationships are realistic: price driven by living area, quality, age.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)

    living_area   = rng.integers(600, 4000, n).astype(float)
    basement_sqft = (living_area * rng.uniform(0.3, 0.9, n)).astype(float)
    garage_sqft   = rng.integers(0, 900, n).astype(float)
    lot_area      = rng.integers(3000, 20000, n).astype(float)
    overall_qual  = rng.integers(1, 11, n).astype(float)
    overall_cond  = rng.integers(1, 11, n).astype(float)
    house_age     = rng.integers(0, 100, n).astype(float)
    full_bath     = rng.integers(1, 4, n).astype(float)
    bedrooms      = rng.integers(1, 6, n).astype(float)
    total_rooms   = bedrooms + rng.integers(2, 6, n).astype(float)
    garage_cap    = rng.integers(0, 4, n).astype(float)
    fireplaces    = rng.integers(0, 3, n).astype(float)

    # Log-price: realistic linear combination + noise
    log_price = (
        9.5
        + 0.0004  * living_area
        + 0.0001  * basement_sqft
        + 0.12    * overall_qual
        - 0.003   * house_age
        + 0.06    * full_bath
        + 0.00001 * lot_area
        + rng.normal(0, 0.15, n)
    )
    sale_price = np.exp(log_price).astype(float)

    return pd.DataFrame({
        TARGET_COL:        sale_price,
        "living_area_sqft": living_area,
        "basement_sqft":    basement_sqft,
        "garage_sqft":      garage_sqft,
        "lot_area":         lot_area,
        "overall_qual":     overall_qual,
        "overall_cond":     overall_cond,
        "house_age_years":  house_age,
        "full_bath":        full_bath,
        "bedrooms":         bedrooms,
        "total_rooms":      total_rooms,
        "garage_capacity":  garage_cap,
        "fireplaces":       fireplaces,
    })


def make_dataloaders(cfg: dict):
    """
    Returns (train_loader, val_loader, scaler, y_scaler, feature_names).
    Target is log-transformed to reduce skew; we report both log-space
    and original-space metrics.
    """
    try:
        df = _load_bigquery()
    except Exception as exc:
        print(f"[INFO] BigQuery unavailable ({exc}). Using synthetic fallback.")
        df = _make_synthetic()

    # Fill remaining NaN with column median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[TARGET_COL].values.astype(np.float32)

    # Log-transform target (predicting log-price improves linearity)
    y = np.log1p(y_raw).astype(np.float32)

    print(f"[Data] Samples={len(X):,}  Features={X.shape[1]}")
    print(f"       SalePrice range: ${y_raw.min():,.0f} – ${y_raw.max():,.0f}  "
          f"median=${np.median(y_raw):,.0f}")
    print(f"       log(1+SalePrice): mean={y.mean():.3f}  std={y.std():.3f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg["val_frac"], random_state=cfg["seed"])

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_val  = scaler.transform(X_val).astype(np.float32)

    def _loader(Xa, ya, shuffle):
        ds = TensorDataset(torch.tensor(Xa), torch.tensor(ya))
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    return (_loader(X_tr, y_tr, True),
            _loader(X_val, y_val, False),
            scaler, FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model
# ─────────────────────────────────────────────────────────────────────────────
class LinearRegressionModel(nn.Module):
    """
    Multivariate linear regression implemented as a single nn.Linear layer.
    Output is log(1 + SalePrice); invert with expm1() for dollar predictions.

    h_theta(x) = W·x + b
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)   # shape [B]


def build_model(cfg: dict, n_features: int) -> nn.Module:
    return LinearRegressionModel(n_features)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train (one epoch)
# ─────────────────────────────────────────────────────────────────────────────
def train(model: nn.Module,
          loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          device: torch.device) -> float:
    """Train for one epoch; returns mean MSE loss (log-price space)."""
    model.train()
    total_loss, total_n = 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
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
    Returns dict with keys: mse, r2, mae, rmse, mse_dollars, mae_dollars,
                            rmse_dollars (back-transformed to USD).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred  = model(Xb)
            loss  = criterion(pred, yb)
            total_loss += loss.item() * len(yb)
            total_n    += len(yb)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # Log-space metrics
    mse  = float(np.mean((preds - labels) ** 2))
    mae  = float(mean_absolute_error(labels, preds))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(labels, preds))

    # Back-transform to USD (expm1 is inverse of log1p)
    preds_usd  = np.expm1(preds)
    labels_usd = np.expm1(labels)
    mae_usd  = float(mean_absolute_error(labels_usd, preds_usd))
    rmse_usd = float(np.sqrt(np.mean((preds_usd - labels_usd) ** 2)))

    return {
        "loss":        total_loss / total_n,
        "mse":         mse,
        "r2":          r2,
        "mae":         mae,
        "rmse":        rmse,
        "mae_dollars": mae_usd,
        "rmse_dollars": rmse_usd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Predict
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module,
            X: np.ndarray,
            device: torch.device,
            return_dollars: bool = True):
    """
    Returns log-price predictions (and optionally USD prices) for raw
    pre-scaled numpy input.
    """
    model.eval()
    Xt    = torch.tensor(X, dtype=torch.float32).to(device)
    logp  = model(Xt).cpu().numpy()
    if return_dollars:
        return logp, np.expm1(logp)
    return logp


# ─────────────────────────────────────────────────────────────────────────────
# 9. Save artifacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(model: nn.Module,
                   metrics: dict,
                   cfg: dict,
                   out_dir: str = ".") -> None:
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(),
               os.path.join(out_dir, "linreg_housing_weights.pt"))

    def _serial(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    payload = {"cfg": cfg,
               "train_metrics": metrics.get("train", {}),
               "val_metrics":   metrics.get("val", {})}
    with open(os.path.join(out_dir, "linreg_housing_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_serial)

    # Human-readable coefficient table
    feat_names = cfg.get("feature_names", [])
    w = model.linear.weight.data.cpu().numpy().flatten()
    b = model.linear.bias.data.cpu().item()
    lines  = ["Linear Regression Coefficients (log-price space)\n",
               "=" * 56 + "\n",
               "Positive weight → higher price\n",
               "Negative weight → lower price\n\n"]
    ranked = sorted(zip(feat_names, w), key=lambda t: abs(t[1]), reverse=True)
    for name, wi in ranked:
        bar = "█" * int(abs(wi) * 20)
        lines.append(f"  {name:<22} {wi:+.5f}  {bar}\n")
    lines.append(f"\n  {'bias':<22} {b:+.5f}\n")
    with open(os.path.join(out_dir, "linreg_housing_coefficients.txt"), "w") as f:
        f.writelines(lines)

    print(f"[Artifacts] Saved to '{out_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = {
        "seed":         42,
        "val_frac":     0.20,
        "batch_size":   64,
        "lr":           5e-3,
        "weight_decay": 1e-4,
        "epochs":       120,
        "out_dir":      "artifacts/linreg_bigquery_housing",
        # quality gates
        "min_val_r2":       0.70,
        "max_val_mse_log":  0.10,
    }

    meta = get_task_metadata()
    print("=" * 60)
    print(f"Task : {meta['task_id']}")
    print(f"Goal : Predict log(1 + {meta['dataset']['target']})")
    print("=" * 60)

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(cfg["seed"])
    device = get_device()
    print(f"[Device] {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, scaler, feat_names = make_dataloaders(cfg)
    cfg["feature_names"] = feat_names
    n_features = len(feat_names)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model(cfg, n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=False)

    print(f"\n[Model] {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"        Parameters: {total_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>6}  {'Train MSE':>11}  {'Val MSE':>9}  "
          f"{'Val R²':>8}  {'Val MAE ($)':>13}  {'LR':>10}")
    print("-" * 65)

    loss_history, val_loss_history = [], []
    best_val_r2 = -float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss      = train(model, train_loader, optimizer, criterion, device)
        val_metrics  = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["mse"])

        loss_history.append(tr_loss)
        val_loss_history.append(val_metrics["mse"])

        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]

        if epoch % 15 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>6}  {tr_loss:>11.5f}  "
                  f"{val_metrics['mse']:>9.5f}  "
                  f"{val_metrics['r2']:>8.4f}  "
                  f"${val_metrics['mae_dollars']:>11,.0f}  "
                  f"{lr_now:>10.2e}")

    # Verify loss decreased overall
    initial_loss = np.mean(loss_history[:5])
    final_loss   = np.mean(loss_history[-5:])
    loss_decreased = final_loss < initial_loss
    print(f"\n[Training] Loss decreased: {initial_loss:.5f} → {final_loss:.5f}"
          f"  ({'✓' if loss_decreased else '✗'})")

    # ── Final evaluation on both splits ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics_f = evaluate(model, val_loader,   criterion, device)

    for split, m in [("Train", train_metrics), ("Val", val_metrics_f)]:
        print(f"\n[{split}]")
        print(f"  MSE  (log-price)  : {m['mse']:.5f}")
        print(f"  RMSE (log-price)  : {m['rmse']:.5f}")
        print(f"  MAE  (log-price)  : {m['mae']:.5f}")
        print(f"  R²   (log-price)  : {m['r2']:.4f}")
        print(f"  MAE  (USD)        : ${m['mae_dollars']:,.0f}")
        print(f"  RMSE (USD)        : ${m['rmse_dollars']:,.0f}")

    # ── Coefficient table ─────────────────────────────────────────────────────
    w      = model.linear.weight.data.cpu().numpy().flatten()
    b      = model.linear.bias.data.cpu().item()
    ranked = sorted(zip(feat_names, w), key=lambda t: abs(t[1]), reverse=True)
    print("\n[Regression Coefficients  (|magnitude| ranked)]")
    print("  Positive = price increases  |  Negative = price decreases")
    for name, wi in ranked:
        bar = "█" * int(abs(wi) * 20)
        print(f"  {name:<22} {wi:+.5f}  {bar}")
    print(f"  {'bias':<22} {b:+.5f}")

    # ── Sample predictions ────────────────────────────────────────────────────
    # Grab first batch of validation data for spot-check
    sample_X, sample_y = next(iter(val_loader))
    sample_X_np = sample_X.numpy()
    _, pred_usd = predict(model, sample_X_np, device, return_dollars=True)
    true_usd    = np.expm1(sample_y.numpy())

    print("\n[Sample Predictions  (first 8 from validation set)]")
    print(f"  {'Predicted ($)':>15}  {'Actual ($)':>13}  {'Error ($)':>12}")
    print("  " + "-" * 44)
    for p, a in zip(pred_usd[:8], true_usd[:8]):
        err = p - a
        print(f"  ${p:>14,.0f}  ${a:>12,.0f}  {'+' if err>=0 else ''}"
              f"${err:>10,.0f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    save_artifacts(model,
                   {"train": train_metrics, "val": val_metrics_f},
                   cfg,
                   out_dir=cfg["out_dir"])

    # ── Quality assertions (self-verifiable exit) ─────────────────────────────
    print("\n[Quality Gates]")
    val_r2  = val_metrics_f["r2"]
    val_mse = val_metrics_f["mse"]
    passed_r2  = val_r2  >= cfg["min_val_r2"]
    passed_mse = val_mse <= cfg["max_val_mse_log"]
    passed_mono = loss_decreased

    print(f"  Val R²  {val_r2:.4f} >= {cfg['min_val_r2']}   "
          f"→ {'✓ PASS' if passed_r2 else '✗ FAIL'}")
    print(f"  Val MSE {val_mse:.5f} <= {cfg['max_val_mse_log']}  "
          f"→ {'✓ PASS' if passed_mse else '✗ FAIL'}")
    print(f"  Loss decreased overall           "
          f"→ {'✓ PASS' if passed_mono else '✗ FAIL'}")
    print(f"  Best Val R² across training: {best_val_r2:.4f}")

    assert passed_r2, \
        f"FAIL: val R² {val_r2:.4f} < threshold {cfg['min_val_r2']}"
    assert passed_mse, \
        f"FAIL: val MSE {val_mse:.5f} > threshold {cfg['max_val_mse_log']}"
    assert passed_mono, \
        "FAIL: training loss did not decrease — model may not have converged."

    print("\n[Exit] exit_code=0  ALL QUALITY GATES PASSED ✓")
    sys.exit(0)
