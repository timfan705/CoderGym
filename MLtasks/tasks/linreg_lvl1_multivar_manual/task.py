"""
Multivariate Linear Regression using Raw PyTorch Tensors — Synthetic Energy Dataset

Mathematical Formulation:
- Hypothesis (vector form): h_theta(X) = X_aug @ theta
    where X_aug is X with a bias column of 1s prepended
    and theta = [theta_0, theta_1, theta_2, theta_3]
- Cost Function (MSE): J(theta) = (1/2m) * ||X_aug @ theta - y||^2
- Gradient: grad = (1/m) * X_aug^T @ (X_aug @ theta - y)
- Update: theta = theta - lr * grad

This implementation uses ONLY PyTorch tensors without torch.nn, torch.optim, or autograd.
Dataset: Synthetic energy use: y = 2*x1 + 5*x2 - 1.5*x3 + 30 + noise
    x1 = outdoor temperature (°C)
    x2 = number of occupants
    x3 = hour of day
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure output directory exists
OUTPUT_DIR = './output/tasks/linreg_lvl1_multivar_manual'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'linreg_lvl1_multivar_manual',
        'description': 'Multivariate Linear Regression on synthetic energy dataset using raw PyTorch tensors',
        'input_dim': 3,
        'output_dim': 1,
        'model_type': 'linear_regression',
        'loss_type': 'mse',
        'optimization': 'batch_gradient_descent',
        'true_theta_0': 30.0,
        'true_theta_1': 2.0,
        'true_theta_2': 5.0,
        'true_theta_3': -1.5,
        'features': ['outdoor_temp', 'occupants', 'hour_of_day']
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=300, train_ratio=0.8, noise_std=5.0, batch_size=32):
    """
    Create synthetic energy use dataset with 3 features.
    True relationship: y = 2*x1 + 5*x2 - 1.5*x3 + 30 + noise

    Args:
        n_samples: Number of samples to generate
        train_ratio: Ratio of training data
        noise_std: Standard deviation of noise
        batch_size: Batch size for dataloaders

    Returns:
        train_loader, val_loader, X_train_aug, X_val_aug, y_train, y_val
        (X includes bias column already appended)
    """
    # Generate features
    x1 = np.random.uniform(0, 35, n_samples)     # outdoor temp  0–35 °C
    x2 = np.random.uniform(1, 10, n_samples)     # occupants     1–10
    x3 = np.random.uniform(0, 23, n_samples)     # hour of day   0–23
    y  = 2.0 * x1 + 5.0 * x2 - 1.5 * x3 + 30.0 + np.random.normal(0, noise_std, n_samples)

    # Stack features and standardize
    X = np.stack([x1, x2, x3], axis=1)           # (n_samples, 3)
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_norm)          # (n_samples, 3)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)  # (n_samples, 1)

    # Prepend bias column of 1s -> (n_samples, 4)
    ones     = torch.ones(n_samples, 1)
    X_aug    = torch.cat([ones, X_tensor], dim=1)

    # Train / val split
    n_train = int(n_samples * train_ratio)
    X_train, X_val = X_aug[:n_train], X_aug[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

    # Dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset   = torch.utils.data.TensorDataset(X_val,   y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


class MultivarLinearRegressionRaw:
    """
    Multivariate Linear Regression from scratch using raw PyTorch tensors.

    Hypothesis (vector form): h_theta(X) = X_aug @ theta
        X_aug shape: (N, n_features+1)  — first column is all 1s (bias)
        theta shape: (n_features+1, 1)

    Gradient: grad = (1/m) * X_aug^T @ (X_aug @ theta - y)
    Update:   theta = theta - lr * grad

    No torch.nn, torch.optim, or autograd used anywhere.
    """

    def __init__(self, n_features=3, device=None):
        self.device     = device if device is not None else get_device()
        self.n_features = n_features
        # theta shape: (n_features+1, 1) — includes bias
        self.theta = torch.zeros(n_features + 1, 1, requires_grad=False).to(self.device)

    def forward(self, X_aug):
        """
        Forward pass: h_theta(X) = X_aug @ theta

        Args:
            X_aug: Augmented input tensor of shape (N, n_features+1)
        Returns:
            Predictions of shape (N, 1)
        """
        return X_aug @ self.theta

    def compute_loss(self, y_pred, y_true):
        """
        Compute MSE loss.

        MSE = (1/2m) * ||y_pred - y_true||^2

        Args:
            y_pred: Predictions of shape (N, 1)
            y_true: Ground truth of shape (N, 1)
        Returns:
            MSE loss scalar
        """
        return torch.mean((y_pred - y_true) ** 2) / 2

    def compute_gradients(self, y_pred, y_true, X_aug):
        """
        Compute gradients manually using matrix form.

        grad = (1/m) * X_aug^T @ (y_pred - y_true)

        Args:
            y_pred: Predictions of shape (N, 1)
            y_true: Ground truth of shape (N, 1)
            X_aug: Augmented input of shape (N, n_features+1)
        Returns:
            grad tensor of shape (n_features+1, 1)
        """
        m      = float(y_true.shape[0])
        errors = y_pred - y_true                  # (N, 1)
        grad   = (1.0 / m) * (X_aug.t() @ errors) # (n_features+1, 1)
        return grad

    def update_parameters(self, grad, lr):
        """
        Update theta using gradient descent.

        theta = theta - lr * grad

        Args:
            grad: Gradient tensor of shape (n_features+1, 1)
            lr: Learning rate
        """
        with torch.no_grad():
            self.theta -= lr * grad

    def fit(self, train_loader, val_loader=None, epochs=1000, lr=0.01, verbose=True):
        """
        Train the model using manual batch gradient descent.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Whether to print progress

        Returns:
            dict with loss_history and val_loss_history
        """
        loss_history     = []
        val_loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.forward(X_batch)
                loss   = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss.item()
                n_batches  += 1

                grad = self.compute_gradients(y_pred, y_batch, X_batch)
                self.update_parameters(grad, lr)

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

        Computes MSE and R2 score. Also reports learned theta values.

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
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred  = self.forward(X_batch)
                all_preds.append(y_pred)
                all_targets.append(y_batch)

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)

        mse    = torch.mean((y_pred - y_true) ** 2).item()
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        r2     = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        theta_vals = self.theta.squeeze().tolist()

        metrics = {
            'mse':     mse,
            'r2':      r2,
            'theta_0': theta_vals[0],
            'theta_1': theta_vals[1],
            'theta_2': theta_vals[2],
            'theta_3': theta_vals[3],
        }

        if return_dict:
            return metrics
        return mse

    def predict(self, X_aug):
        """
        Make predictions on new augmented data.

        Args:
            X_aug: Augmented input tensor or array of shape (N, n_features+1)
        Returns:
            Predictions tensor of shape (N, 1)
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(X_aug, torch.Tensor):
                X_aug = torch.FloatTensor(X_aug)
            X_aug = X_aug.to(self.device)
            return self.forward(X_aug)

    def eval(self):
        """Set model to evaluation mode (no-op for raw tensor model)."""
        pass

    def state_dict(self):
        """Return model state for saving."""
        return {'theta': self.theta}

    def load_state_dict(self, state_dict):
        """Load model state."""
        self.theta = state_dict['theta']


def build_model(n_features=3, device=None):
    """Build and return a MultivarLinearRegressionRaw model."""
    return MultivarLinearRegressionRaw(n_features=n_features, device=device)


def train(model, train_loader, val_loader, epochs=1000, lr=0.01):
    """Train the model and return history."""
    return model.fit(train_loader, val_loader, epochs=epochs, lr=lr)


def evaluate(model, data_loader):
    """Evaluate model and return metrics dict."""
    return model.evaluate(data_loader, return_dict=True)


def predict(model, X_aug):
    """Return predictions for augmented input X_aug."""
    return model.predict(X_aug)


def save_artifacts(model, history, output_dir=OUTPUT_DIR):
    """
    Save model state and training plots.

    Args:
        model: Trained MultivarLinearRegressionRaw model
        history: dict with loss_history and val_loss_history
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model parameters
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))

    # Plot training and validation loss
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss_history'],     label='Train Loss')
    plt.plot(history['val_loss_history'], label='Val Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve — Multivariate Linear Regression')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linreg_lvl1_multivar_loss.png'))
    plt.close()

    print(f'Artifacts saved to {output_dir}')


if __name__ == '__main__':
    # ── Setup ──────────────────────────────────────────────────────
    set_seed(42)
    device = get_device()
    meta   = get_task_metadata()
    print('Task:', meta['task_name'])
    print('Device:', device)
    print(f"True params: theta_0={meta['true_theta_0']}  theta_1={meta['true_theta_1']}  "
          f"theta_2={meta['true_theta_2']}  theta_3={meta['true_theta_3']}")
    print('(theta_1–3 are in normalised-feature space after training)\n')

    # ── Data ───────────────────────────────────────────────────────
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=300, train_ratio=0.8, noise_std=5.0, batch_size=32
    )
    print(f'Train samples: {len(X_train)}  |  Val samples: {len(X_val)}')

    # ── Build & Train ──────────────────────────────────────────────
    model   = build_model(n_features=3, device=device)
    history = train(model, train_loader, val_loader, epochs=1000, lr=0.05)

    # ── Evaluate ───────────────────────────────────────────────────
    train_metrics = evaluate(model, train_loader)
    val_metrics   = evaluate(model, val_loader)

    print('\n── Train Metrics ──')
    print(f"  MSE : {train_metrics['mse']:.4f}")
    print(f"  R2  : {train_metrics['r2']:.4f}")

    print('\n── Validation Metrics ──')
    print(f"  MSE : {val_metrics['mse']:.4f}")
    print(f"  R2  : {val_metrics['r2']:.4f}")

    print('\n── Learned Parameters ──')
    feature_names = ['bias', 'temp_norm', 'occupants_norm', 'hour_norm']
    for i, name in enumerate(feature_names):
        print(f"  theta_{i} ({name}) = {val_metrics[f'theta_{i}']:.4f}")

    # ── Save Artifacts ─────────────────────────────────────────────
    save_artifacts(model, history)

    # ── Assertions ─────────────────────────────────────────────────
    assert val_metrics['r2'] > 0.9, \
        f"FAIL: val R2={val_metrics['r2']:.4f} is below threshold 0.9"
    assert val_metrics['mse'] < 50.0, \
        f"FAIL: val MSE={val_metrics['mse']:.4f} is above threshold 50.0"
    assert history['loss_history'][-1] < history['loss_history'][0], \
        'FAIL: training loss did not decrease'
    assert val_metrics['theta_0'] > 0, \
        f"FAIL: bias theta_0={val_metrics['theta_0']:.4f} should be positive"

    print('\n✓ All assertions passed.')
    sys.exit(0)