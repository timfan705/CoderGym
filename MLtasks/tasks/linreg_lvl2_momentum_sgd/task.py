"""
Linear Regression using torch.optim — SGD vs SGD with Momentum Comparison

Mathematical Formulation:
- Hypothesis: h_theta(x) = W * x + b   (torch.nn.Linear)
- Cost Function (MSE): J(theta) = (1/2m) * sum((h_theta(x_i) - y_i)^2)

SGD update:
    theta_{t+1} = theta_t - lr * grad_t

SGD with Momentum (Polyak):
    v_{t+1}     = momentum * v_t + grad_t
    theta_{t+1} = theta_t - lr * v_{t+1}

Momentum accumulates a velocity vector in directions of persistent gradient,
which typically leads to faster convergence and smoother loss curves.

This task trains TWO models side-by-side on the same data with the same
initialisation — one with plain SGD and one with SGD + momentum=0.9.

Dataset: Synthetic 4-feature regression
    y = 3*x1 - 2*x2 + 1.5*x3 + 0*x4 + 5 + noise
    (x4 is pure noise — a sanity check that the model doesn't overfit to it)
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure output directory exists
OUTPUT_DIR = './output/tasks/linreg_lvl2_momentum_sgd'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    """Return metadata about the task."""
    return {
        'task_name': 'linreg_lvl2_momentum_sgd',
        'description': 'Linear Regression comparing SGD vs SGD+Momentum using torch.optim',
        'input_dim': 4,
        'output_dim': 1,
        'model_type': 'linear_regression',
        'loss_type': 'mse',
        'optimization': 'sgd_vs_momentum_comparison',
        'true_weights': [3.0, -2.0, 1.5, 0.0],
        'true_bias': 5.0
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=500, train_ratio=0.8, noise_std=2.0, batch_size=32):
    """
    Create synthetic 4-feature regression dataset.
    True relationship: y = 3*x1 - 2*x2 + 1.5*x3 + 0*x4 + 5 + noise
    x4 is pure noise — weight should converge near zero.

    Args:
        n_samples: Number of samples to generate
        train_ratio: Ratio of training data
        noise_std: Standard deviation of noise
        batch_size: Batch size for dataloaders

    Returns:
        train_loader, val_loader, X_train, X_val, y_train, y_val
    """
    # Generate features and target
    X        = np.random.randn(n_samples, 4)
    true_w   = np.array([3.0, -2.0, 1.5, 0.0])
    y        = X @ true_w + 5.0 + np.random.normal(0, noise_std, n_samples)

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_norm)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)   # (n_samples, 1)

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


class LinearRegressionOptim(nn.Module):
    """
    Linear Regression model using torch.nn.Linear + torch.optim.

    Hypothesis: h_theta(x) = W * x + b
    Supports any torch.optim optimizer passed in at training time.
    """

    def __init__(self, input_dim=4):
        super(LinearRegressionOptim, self).__init__()
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

    def fit(self, train_loader, val_loader=None, epochs=200, lr=0.05,
            momentum=0.0, label='SGD', verbose=True):
        """
        Train using torch.optim.SGD with optional momentum.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            lr: Learning rate
            momentum: Momentum factor (0.0 = plain SGD)
            label: Label string for print output
            verbose: Whether to print progress

        Returns:
            dict with loss_history and val_loss_history
        """
        loss_fn   = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        loss_history     = []
        val_loss_history = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss   = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader, return_dict=False)
                val_loss_history.append(val_loss)

            if verbose and (epoch + 1) % 50 == 0:
                print(f'[{label}] Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.6f}')

        return {
            'loss_history':     loss_history,
            'val_loss_history': val_loss_history
        }

    def evaluate(self, data_loader, return_dict=True):
        """
        Evaluate the model on a data loader.

        Computes MSE and R2 score. Also reports learned weights.

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

        weights = self.linear.weight.data.squeeze().tolist()
        bias    = self.linear.bias.data.item()

        metrics = {
            'mse':  mse,
            'r2':   r2,
            'weights': weights,
            'bias':    bias
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


def build_model(input_dim=4, device=None):
    """Build and return a LinearRegressionOptim model."""
    if device is None:
        device = get_device()
    return LinearRegressionOptim(input_dim=input_dim).to(device)


def train(model, train_loader, val_loader, epochs=200, lr=0.05, momentum=0.0, label='SGD'):
    """Train the model with given optimizer settings and return history."""
    return model.fit(train_loader, val_loader,
                     epochs=epochs, lr=lr, momentum=momentum, label=label)


def evaluate(model, data_loader):
    """Evaluate model and return metrics dict."""
    return model.evaluate(data_loader, return_dict=True)


def predict(model, X):
    """Return predictions for input X."""
    return model.predict(X)


def save_artifacts(histories, output_dir=OUTPUT_DIR):
    """
    Save comparison training plot for SGD vs SGD+Momentum.

    Args:
        histories: dict mapping label -> history dict
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for label, history in histories.items():
        axes[0].plot(history['loss_history'],     label=label)
        axes[1].plot(history['val_loss_history'], label=label, linestyle='--')

    for ax, title in zip(axes, ['Train MSE', 'Val MSE']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(title)
        ax.legend()

    plt.suptitle('SGD vs SGD+Momentum — Linear Regression')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linreg_lvl2_momentum_comparison.png'))
    plt.close()

    print(f'Artifacts saved to {output_dir}')


if __name__ == '__main__':
    # ── Setup ──────────────────────────────────────────────────────
    set_seed(42)
    device = get_device()
    meta   = get_task_metadata()
    print('Task:', meta['task_name'])
    print('Device:', device)
    print(f"True weights: {meta['true_weights']}  bias: {meta['true_bias']}")
    print('(x4 is pure noise — its weight should stay near zero)\n')

    # ── Data ───────────────────────────────────────────────────────
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        n_samples=500, train_ratio=0.8, noise_std=2.0, batch_size=32
    )
    print(f'Train samples: {len(X_train)}  |  Val samples: {len(X_val)}')

    EPOCHS = 200
    LR     = 0.05

    # ── Model A: Plain SGD ─────────────────────────────────────────
    print('\n── Training Model A: Plain SGD ──')
    set_seed(0)
    model_sgd = build_model(input_dim=4, device=device)
    hist_sgd  = train(model_sgd, train_loader, val_loader,
                      epochs=EPOCHS, lr=LR, momentum=0.0, label='SGD')

    # ── Model B: SGD + Momentum ────────────────────────────────────
    print('\n── Training Model B: SGD + Momentum=0.9 ──')
    set_seed(0)   # same init for fair comparison
    model_mom = build_model(input_dim=4, device=device)
    hist_mom  = train(model_mom, train_loader, val_loader,
                      epochs=EPOCHS, lr=LR, momentum=0.9, label='SGD+Momentum')

    # ── Evaluate ───────────────────────────────────────────────────
    train_sgd = evaluate(model_sgd, train_loader)
    val_sgd   = evaluate(model_sgd, val_loader)
    train_mom = evaluate(model_mom, train_loader)
    val_mom   = evaluate(model_mom, val_loader)

    print('\n── Final Metrics ──')
    print(f"SGD          Train MSE={train_sgd['mse']:.4f}  Val MSE={val_sgd['mse']:.4f}  Val R2={val_sgd['r2']:.4f}")
    print(f"SGD+Momentum Train MSE={train_mom['mse']:.4f}  Val MSE={val_mom['mse']:.4f}  Val R2={val_mom['r2']:.4f}")

    print('\n── Learned Weights (normalised scale) ──')
    feature_names = ['x1', 'x2', 'x3', 'x4_noise']
    print(f"  {'Feature':<12} {'SGD':>10} {'Momentum':>12}")
    for i, name in enumerate(feature_names):
        ws = val_sgd['weights'][i]
        wm = val_mom['weights'][i]
        print(f"  {name:<12} {ws:>10.4f} {wm:>12.4f}")

    # Convergence speed: epochs to reach 50% of total loss reduction
    def epochs_to_threshold(loss_list, fraction=0.5):
        start  = loss_list[0]
        end    = loss_list[-1]
        target = start - fraction * (start - end)
        for i, v in enumerate(loss_list):
            if v <= target:
                return i + 1
        return len(loss_list)

    ep_sgd = epochs_to_threshold(hist_sgd['loss_history'])
    ep_mom = epochs_to_threshold(hist_mom['loss_history'])
    print(f'\nEpochs to 50% loss reduction:')
    print(f"  SGD:          {ep_sgd}")
    print(f"  SGD+Momentum: {ep_mom}  {'(faster ✓)' if ep_mom <= ep_sgd else '(similar)'}")

    # ── Save Artifacts ─────────────────────────────────────────────
    save_artifacts({'SGD': hist_sgd, 'SGD+Momentum': hist_mom})

    # ── Assertions ─────────────────────────────────────────────────
    assert val_sgd['r2'] > 0.85, \
        f"FAIL: SGD val R2={val_sgd['r2']:.4f} is below threshold 0.85"
    assert val_mom['r2'] > 0.85, \
        f"FAIL: SGD+Momentum val R2={val_mom['r2']:.4f} is below threshold 0.85"

    # Momentum should converge at least as fast (with 20-epoch grace)
    assert ep_mom <= ep_sgd + 20, \
        f"FAIL: Momentum ({ep_mom} epochs) not faster than plain SGD ({ep_sgd} epochs)"

    # Noise feature weight should be small in both models
    noise_sgd = abs(val_sgd['weights'][3])
    noise_mom = abs(val_mom['weights'][3])
    assert noise_sgd < 1.5, \
        f"FAIL: SGD noise-feature weight too large ({noise_sgd:.4f})"
    assert noise_mom < 1.5, \
        f"FAIL: Momentum noise-feature weight too large ({noise_mom:.4f})"

    print('\n✓ All assertions passed.')
    sys.exit(0)