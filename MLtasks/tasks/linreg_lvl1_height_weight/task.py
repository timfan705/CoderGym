"""
Linear Regression using Raw PyTorch Tensors — Height to Weight Dataset

Mathematical Formulation:
- Hypothesis: h_theta(x) = theta_0 + theta_1 * x
- Cost Function (MSE): J(theta) = (1/2m) * sum((h_theta(x_i) - y_i)^2)
- Gradient Descent Updates:
    dJ/dtheta_0 = (1/m) * sum(h_theta(x_i) - y_i)
    dJ/dtheta_1 = (1/m) * sum((h_theta(x_i) - y_i) * x_i)
    theta = theta - lr * grad(theta)

This implementation uses ONLY PyTorch tensors without torch.nn, torch.optim, or autograd.
Dataset: Synthetic height -> weight relationship: weight = 0.5 * height + 10 + noise
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure output directory exists
OUTPUT_DIR = './output/tasks/linreg_lvl1_height_weight'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'linreg_lvl1_height_weight',
        'description': 'Univariate Linear Regression on synthetic height->weight data using raw PyTorch tensors',
        'input_dim': 1,
        'output_dim': 1,
        'model_type': 'linear_regression',
        'loss_type': 'mse',
        'optimization': 'gradient_descent',
        'true_theta_0': 10.0,
        'true_theta_1': 0.5
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=200, train_ratio=0.8, noise_std=3.0, batch_size=32):
    """
    Create synthetic height -> weight dataset.
    True relationship: weight = 0.5 * height + 10 + noise

    Args:
        n_samples: Number of samples to generate
        train_ratio: Ratio of training data
        noise_std: Standard deviation of noise
        batch_size: Batch size for dataloaders

    Returns:
        train_loader, val_loader, X_train, X_val, y_train, y_val
    """
    # Generate synthetic data: weight = 0.5 * height + 10 + noise
    # Heights in cm ranging roughly 150-195
    height = np.random.uniform(150, 195, n_samples)
    weight = 0.5 * height + 10.0 + np.random.normal(0, noise_std, n_samples)

    # Convert to tensors
    X_tensor = torch.FloatTensor(height).unsqueeze(1)   # Shape: (n_samples, 1)
    y_tensor = torch.FloatTensor(weight).unsqueeze(1)   # Shape: (n_samples, 1)

    # Normalize X for stable gradient descent
    X_mean = X_tensor.mean()
    X_std  = X_tensor.std()
    X_tensor = (X_tensor - X_mean) / X_std

    # Split into train and validation
    n_train = int(n_samples * train_ratio)
    X_train, X_val = X_tensor[:n_train], X_tensor[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset   = torch.utils.data.TensorDataset(X_val,   y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


class LinearRegressionRaw:
    """
    Univariate Linear Regression implemented from scratch using raw PyTorch tensors.

    Hypothesis: h_theta(x) = theta_0 + theta_1 * x
    Parameters:
        - theta_0: bias term (intercept)
        - theta_1: weight (slope)

    No torch.nn, torch.optim, or autograd used anywhere.
    """

    def __init__(self, device=None):
        self.device = device if device is not None else get_device()
        # Initialize parameters to zero
        self.theta_0 = torch.zeros(1, requires_grad=False).to(self.device)
        self.theta_1 = torch.zeros(1, requires_grad=False).to(self.device)

    def forward(self, X):
        """
        Forward pass: h_theta(x) = theta_0 + theta_1 * x

        Args:
            X: Input tensor of shape (N, 1)
        Returns:
            Predictions of shape (N, 1)
        """
        return self.theta_0 + self.theta_1 * X

    def compute_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error loss.

        MSE = (1/2m) * sum((y_pred - y_true)^2)

        Args:
            y_pred: Predictions of shape (N, 1)
            y_true: Ground truth of shape (N, 1)
        Returns:
            MSE loss scalar
        """
        return torch.mean((y_pred - y_true) ** 2) / 2

    def compute_gradients(self, y_pred, y_true, X):
        """
        Compute gradients manually — no autograd.

        dJ/dtheta_0 = (1/m) * sum(y_pred - y_true)
        dJ/dtheta_1 = (1/m) * sum((y_pred - y_true) * x)

        Args:
            y_pred: Predictions of shape (N, 1)
            y_true: Ground truth of shape (N, 1)
            X: Input of shape (N, 1)
        Returns:
            grad_theta_0, grad_theta_1
        """
        errors = y_pred - y_true
        grad_theta_0 = torch.mean(errors)
        grad_theta_1 = torch.mean(errors * X)
        return grad_theta_0, grad_theta_1

    def update_parameters(self, grad_theta_0, grad_theta_1, lr):
        """
        Update parameters using gradient descent.

        theta = theta - lr * grad(theta)

        Args:
            grad_theta_0: Gradient for bias
            grad_theta_1: Gradient for weight
            lr: Learning rate
        """
        with torch.no_grad():
            self.theta_0 -= lr * grad_theta_0
            self.theta_1 -= lr * grad_theta_1

    def fit(self, train_loader, val_loader=None, epochs=1000, lr=0.01, verbose=True):
        """
        Train the model using manual gradient descent.

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

                grad_0, grad_1 = self.compute_gradients(y_pred, y_batch, X_batch)
                self.update_parameters(grad_0, grad_1, lr)

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

        Computes MSE, R2 score, and parameter accuracy vs true values
        (true theta_0 = 10.0, true theta_1 = 0.5 in normalized-x space).

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

        # MSE
        mse = torch.mean((y_pred - y_true) ** 2).item()

        # R2 score
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Parameter errors (theta_0 true=~82.5 in original scale; we report learned values)
        theta_0_val = self.theta_0.item()
        theta_1_val = self.theta_1.item()

        metrics = {
            'mse':    mse,
            'r2':     r2,
            'theta_0': theta_0_val,
            'theta_1': theta_1_val,
        }

        if return_dict:
            return metrics
        return mse

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input tensor or array of shape (N, 1)
        Returns:
            Predictions tensor of shape (N, 1)
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            return self.forward(X)

    def eval(self):
        """Set model to evaluation mode (no-op for raw tensor model)."""
        pass

    def state_dict(self):
        """Return model state for saving."""
        return {
            'theta_0': self.theta_0,
            'theta_1': self.theta_1
        }

    def load_state_dict(self, state_dict):
        """Load model state."""
        self.theta_0 = state_dict['theta_0']
        self.theta_1 = state_dict['theta_1']


def build_model(device=None):
    """Build and return a LinearRegressionRaw model."""
    return LinearRegressionRaw(device=device)


def train(model, train_loader, val_loader, epochs=1000, lr=0.01):
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
        model: Trained LinearRegressionRaw model
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
    plt.title('Training Curve — Height/Weight Linear Regression')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linreg_lvl1_height_weight_loss.png'))
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
        n_samples=200, train_ratio=0.8, noise_std=3.0, batch_size=32
    )
    print(f'Train samples: {len(X_train)}  |  Val samples: {len(X_val)}')

    # ── Build & Train ──────────────────────────────────────────────
    model   = build_model(device=device)
    history = train(model, train_loader, val_loader, epochs=1000, lr=0.01)

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
    print(f"  theta_0 (bias)  = {val_metrics['theta_0']:.4f}")
    print(f"  theta_1 (slope) = {val_metrics['theta_1']:.4f}")
    print(f"  (Note: theta_1 is in normalised-x space)")

    # ── Save Artifacts ─────────────────────────────────────────────
    save_artifacts(model, history)

    # ── Assertions ─────────────────────────────────────────────────
    assert val_metrics['r2'] > 0.9, \
        f"FAIL: val R2={val_metrics['r2']:.4f} is below threshold 0.9"
    assert val_metrics['mse'] < 20.0, \
        f"FAIL: val MSE={val_metrics['mse']:.4f} is above threshold 20.0"
    assert history['loss_history'][-1] < history['loss_history'][0], \
        'FAIL: training loss did not decrease'

    print('\n✓ All assertions passed.')
    sys.exit(0)