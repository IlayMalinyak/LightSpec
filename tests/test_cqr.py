import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from LightSpec.nn.optim import CQR

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Create synthetic data
def generate_data(n_samples=1000, n_labels=2):
    # Generate features
    X = np.random.normal(0, 1, (n_samples, 2))

    # Generate labels with heteroscedastic noise
    y = np.zeros((n_samples, n_labels))
    for i in range(n_labels):
        # Create non-linear relationship
        y[:, i] = 2 * X[:, 0] ** 2 + X[:, 1] + np.random.normal(0, 0.5 + np.abs(X[:, 0]), n_samples)

    return X, y


# Simple neural network for quantile regression
class QuantileNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_quantiles):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim * n_quantiles)
        )
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles

    def forward(self, x):
        out = self.network(x)
        # Reshape output to (batch_size, output_dim, n_quantiles)
        return out.view(-1, self.output_dim, self.n_quantiles)


# Function to evaluate coverage
def evaluate_coverage(predictions, targets, quantiles):
    n_quantile_pairs = len(quantiles) // 2
    coverage_stats = []

    for i in range(n_quantile_pairs):
        lower_idx = i
        upper_idx = -(i + 1)

        lower_bound = predictions[:, :, lower_idx]
        upper_bound = predictions[:, :, upper_idx]

        in_interval = np.logical_and(
            targets >= lower_bound,
            targets <= upper_bound
        )

        coverage = np.mean(in_interval)
        expected_coverage = quantiles[upper_idx] - quantiles[lower_idx]
        coverage_stats.append((expected_coverage, coverage))

    return coverage_stats


# Test the implementation
if __name__ == "__main__":
    # Generate data
    X, y = generate_data(n_samples=2000, n_labels=2)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_cal, X_test, y_cal, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_cal = torch.FloatTensor(X_cal)
    y_cal = torch.FloatTensor(y_cal)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Define quantiles
    quantiles = [0.1, 0.5, 0.9]  # Example with 90% prediction interval

    # Initialize model and loss
    model = QuantileNet(input_dim=2, output_dim=2, n_quantiles=len(quantiles))
    criterion = CQR(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train.squeeze())

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Generate predictions for calibration
    model.eval()
    with torch.no_grad():
        cal_preds = model(X_cal).numpy()
        test_preds = model(X_test).numpy()

    # Calibrate and predict
    nc_errors = criterion.calibrate(cal_preds, y_cal.numpy())
    conformal_preds = criterion.predict(test_preds, nc_errors)

    # Evaluate coverage
    coverage_stats = evaluate_coverage(conformal_preds, y_test.numpy(), quantiles)

    # Print results
    print("\nCoverage Statistics:")
    for expected, actual in coverage_stats:
        print(f"Expected coverage: {expected:.2f}, Actual coverage: {actual:.2f}")