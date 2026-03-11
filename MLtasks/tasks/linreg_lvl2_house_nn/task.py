"""
Linear Regression using torch.nn and Autograd — Synthetic House Price Dataset

Mathematical Formulation:
- Hypothesis: h_theta(x) = W * x + b   (torch.nn.Linear)
- Cost Function (MSE): J(theta) = (1/2m) * sum((h_theta(x_i) - y_i)^2)
- Gradient via autograd: loss.backward() computes all gradients automatically
- Manual update (no optimizer): with torch.no_grad(): W -= lr * W.grad

Introduces torch.nn.Linear and loss.backward() while keeping the manual
weight update step to show the bridge between raw tensors and full optimizers.

Dataset: Synthetic house prices with 2 features
    x1 = house size (sq ft, 500–3000)
    x2 = number of bedrooms (1–5)
    y  = 150*x1 + 20000*x2 + 50000 + noise
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure output directory exists
OUTPUT_DIR = './output/tasks/linreg_lvl2_house_nn'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'linreg_lvl2_house_nn',
        'description': 'Linear Regression with torch.nn.Linear and autograd on synthetic house price data',
        'input_dim': 2,
        'output_dim': 1,
        'model_type': 'linear_regression',
        'loss_type': 'mse',
        'optimization': 'manual_update_with_autograd',
        'features': ['house_size_sqft', 'num_bedrooms']
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=400, train_ratio=0.8, noise_std=15000, batch_size=32):
    """
    Create synthetic house price dataset with 2 features.
    True relationship: price = 150*sqft + 20000*beds + 50000 + noise

    Args:
        n_samples: Number of samples to generate
        train_ratio: Ratio of training data
        noise_std: Standard deviation of noise (in dollars)
        batch_size: Batch size for dataloaders

    Returns:
        train_loader, val_loader, X_train, X_val, y_train, y_val
    """
    # Generate features
    sqft = np.random.uniform(500, 3000, n_samples)
    beds = np.random.randint(1, 6, n_samples).astype(float)
    y    = 150.0 * sqft + 20000.0 * beds + 50000.0 + np.random.normal(0, noise_std, n_samples)

    # Stack and standardize features
    X      = np.stack([sqft, beds], axis=1)   # (n_samples, 2)
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Standardize target too
    y_mean = y.mean()
    y_std  = y.std() + 1e-8
    y_norm = (y - y_mean) / y_std

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_norm)
    y_tensor = torch.FloatTensor(y_norm).unsqueeze(1)  # (n_samples, 1)

    # Train / val split
    n_train = int(n_samples * train_ratio)
    X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

    # Dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset   = torch.utils.data.TensorDataset(X_val,   y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


class HousePriceLinearModel(nn.Module):
    """
    Linear Regression model using torch.nn.Linear.

    Hypothesis: h_theta(x) = W * x + b
    Uses autograd for gradient computation but updates weights manually
    inside torch.no_grad() — no torch.optim used.
    """

    def __init__(self, input_dim=2):
        super(HousePriceLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, X):
        """
        Forward pass through nn.Linear.

        Args:
            X: Input tensor of shape (N, input_dim)
        Returns:
            Predictions of shape (N, 1)
        """
        return self.linear(X)

    def fit(self, train_loader, val_loader=None, epochs=500, lr=0.05, verbose=True):
        """
        Train using loss.backward() for gradients and manual weight updates.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Whether to print progress

        Returns:
            dict with loss_history and val_loss_history
        """
        loss_fn          = nn.MSELoss()
        loss_history     = []
        val_loss_history = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in train_loader:
                # Zero gradients manually
                if self.linear.weight.grad is not None:
                    self.linear.weight.grad.zero_()
                if self.linear.bias.grad is not None:
                    self.linear.bias.grad.zero_()

                # Forward pass
                y_pred = self.forward(X_batch)
                loss   = loss_fn(y_pred, y_batch)

                # Backward pass — autograd fills .grad
                loss.backward()

                # Manual weight update — no torch.optim
                with torch.no_grad():
                    self.linear.weight -= lr * self.linear.weight.grad
                    self.linear.bias   -= lr * self.linear.bias.grad

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader, return_dict=False)
                val_loss_history.append(val_loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.6f}')

        return {
            'loss_history':     loss_history,
            'val_loss_history': val_loss_history
        }

    def evaluate(self, data_loader, return_dict=True):
        """
        Evaluate the model on a data loader.

        Computes MSE and R2 score on the given split.

        Args:
            data_loader: Data loader to evaluate on
            return_dict: If True return metrics dict, else return MSE float
        Returns:
            dict with metrics or float (MSE)
        """
        self.eval()

        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                y_pred = self.forward(X_batch)
                all_preds.append(y_pred)
                all_targets.append(y_batch)

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)

        mse    = torch.mean((y_pred - y_true) ** 2).item()
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        r2     = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        weight = self.linear.weight.data.squeeze().tolist()
        bias   = self.linear.bias.data.item()

        metrics = {
            'mse':    mse,
            'r2':     r2,
            'weight_sqft': weight[0],
            'weight_beds': weight[1],
            'bias':        bias
        }

        if return_dict:
            return metrics
        return mse

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input tensor or array of shape (N, input_dim)
        Returns:
            Predictions tensor of shape (N, 1)
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            return self.forward(X)


def build_model(input_dim=2, device=None):
    """Build and return a HousePriceLinearModel."""
    if device is None:
        device = get_device()
    return HousePriceLinearModel(input_dim=input_dim).to(device)


def train(model, train_loader, val_loader, epochs=500, lr=0.05):
    """Train the model and return history."""
    return model.fit(train_loader, val_loader, epochs=epochs, lr=lr)


def evaluate(model, data_loader):
    """Evaluate model and return metrics dict."""
    return model.evaluate(data_loader, return_dict=True)


def predict(model, X):
    """Return predictions for input X."""
    return model.predict(X)


def save_artifacts(model, history, output_dir=OUTPUT_DIR):
    """
    Save model state and training plots.

    Args:
        model: Trained HousePriceLinearModel
        history: dict with loss_history and val_loss_history
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

    # Training loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss_history'],     label='Train Loss')
    plt.plot(history['val_loss_history'], label='Val Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (normalised scale)')
    plt.title('Training Curve — House Price Linear Regression')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linreg_lvl2_house_loss.png'))
    plt.close()

    print(f'Artifacts saved to {output_dir}')


if __name__ == '__main__':
    # ── Setup ──────────────────────────────────────────────────────
    set_seed(42)
    device = get_device()
    print('Task:', get_task_metadata()['task_name'])
    print('Device:', device)

    # ── Data ───────────────────────────────────────────────────────
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=400, train_ratio=0.8, noise_std=15000, batch_size=32
    )
    print(f'Train samples: {len(X_train)}  |  Val samples: {len(X_val)}')

    # ── Build & Train ──────────────────────────────────────────────
    model   = build_model(input_dim=2, device=device)
    history = train(model, train_loader, val_loader, epochs=500, lr=0.05)

    # ── Evaluate ───────────────────────────────────────────────────
    train_metrics = evaluate(model, train_loader)
    val_metrics   = evaluate(model, val_loader)

    print('\n── Train Metrics ──')
    print(f"  MSE : {train_metrics['mse']:.4f}")
    print(f"  R2  : {train_metrics['r2']:.4f}")

    print('\n── Validation Metrics ──')
    print(f"  MSE : {val_metrics['mse']:.4f}")
    print(f"  R2  : {val_metrics['r2']:.4f}")

    print('\n── Learned Parameters (normalised scale) ──')
    print(f"  weight_sqft = {val_metrics['weight_sqft']:.4f}")
    print(f"  weight_beds = {val_metrics['weight_beds']:.4f}")
    print(f"  bias        = {val_metrics['bias']:.4f}")

    # ── Save Artifacts ─────────────────────────────────────────────
    save_artifacts(model, history)

    # ── Assertions ─────────────────────────────────────────────────
    assert val_metrics['r2'] > 0.9, \
        f"FAIL: val R2={val_metrics['r2']:.4f} is below threshold 0.9"
    assert val_metrics['mse'] < 0.2, \
        f"FAIL: val MSE={val_metrics['mse']:.4f} is above threshold 0.2 (normalised scale)"

    first_10 = sum(history['loss_history'][:10])  / 10
    last_10  = sum(history['loss_history'][-10:]) / 10
    assert last_10 < first_10, \
        f'FAIL: training loss did not decrease (first10={first_10:.4f}, last10={last_10:.4f})'

    print('\n✓ All assertions passed.')
    sys.exit(0)