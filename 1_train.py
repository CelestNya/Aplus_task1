"""
1_train.py - SVM Training with SGD Optimization
Implements linear SVM using hinge loss + L2 regularization
"""

import os
import pickle
import numpy as np
import pandas as pd


def train_SVM():
    """Train raw SVM without feature standardization using SGD."""
    # Load data
    data = pd.read_csv('data/train.csv')
    X = data[['Feature1', 'Feature2', 'Feature3']].values
    y_labels = data['Label'].values
    # Convert labels: 0 -> -1, 1 -> 1 (SVM standard form)
    y = np.where(y_labels == 0, -1, 1)

    # Initialize parameters
    n_samples, n_features = X.shape
    C = 1.0  # Regularization parameter
    lr = 0.01  # Learning rate
    epochs = 2000
    seed = 42

    np.random.seed(seed)
    w = np.random.randn(n_features) * 0.01  # Initialize weights
    b = 0.0  # Bias term

    # SGD optimization
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(n_samples):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            # Compute decision function
            decision = yi * (np.dot(w, xi) + b)

            # Hinge loss gradient
            if decision < 1:
                # Misclassified or within margin: apply gradient
                w_grad = w - C * yi * xi
                b_grad = -C * yi
            else:
                # Correctly classified outside margin: regularized loss = 0
                w_grad = w
                b_grad = 0

            # Update weights and bias
            w = w - lr * w_grad
            b = b - lr * b_grad

        # Learning rate decay (optional)
        if epoch % 500 == 0:
            lr *= 0.9

    # Save model
    os.makedirs('model', exist_ok=True)
    model = {'w': w, 'b': b}
    with open('model/svm_raw.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Print hyperplane equation
    print("=" * 60)
    print("train_SVM() - Raw SVM (No Standardization)")
    print("=" * 60)
    print(f"Model saved to: model/svm_raw.pkl")
    print(f"Hyperplane equation:")
    print(f"  {w[0]:.6f}*x + {w[1]:.6f}*y + {w[2]:.6f}*z + {b:.6f} = 0")
    print(f"Weights (w): {w}")
    print(f"Bias (b): {b}")
    print("=" * 60)

    return model


def train_std_SVM():
    """Train SVM with standardized features (zero mean, unit variance)."""
    # Load data
    data = pd.read_csv('data/train.csv')
    X = data[['Feature1', 'Feature2', 'Feature3']].values
    y_labels = data['Label'].values
    # Convert labels: 0 -> -1, 1 -> 1 (SVM standard form)
    y = np.where(y_labels == 0, -1, 1)

    # Standardize features: (X - mean) / std
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0
    X_std = (X - mean) / std

    # Initialize parameters
    n_samples, n_features = X_std.shape
    C = 1.0  # Regularization parameter
    lr = 0.01  # Learning rate
    epochs = 2000
    seed = 42

    np.random.seed(seed)
    w = np.random.randn(n_features) * 0.01  # Initialize weights
    b = 0.0  # Bias term

    # SGD optimization
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X_std[indices]
        y_shuffled = y[indices]

        for i in range(n_samples):
            xi = X_shuffled[i]
            yi = y_shuffled[i]

            # Compute decision function
            decision = yi * (np.dot(w, xi) + b)

            # Hinge loss gradient
            if decision < 1:
                # Misclassified or within margin: apply gradient
                w_grad = w - C * yi * xi
                b_grad = -C * yi
            else:
                # Correctly classified outside margin: regularized loss = 0
                w_grad = w
                b_grad = 0

            # Update weights and bias
            w = w - lr * w_grad
            b = b - lr * b_grad

        # Learning rate decay (optional)
        if epoch % 500 == 0:
            lr *= 0.9

    # Save model with standardization parameters
    os.makedirs('model', exist_ok=True)
    model = {'w': w, 'b': b, 'mean': mean, 'std': std}
    with open('model/svm_std.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Print hyperplane equation
    print("=" * 60)
    print("train_std_SVM() - SVM with Feature Standardization")
    print("=" * 60)
    print(f"Model saved to: model/svm_std.pkl")
    print(f"Feature standardization: mean={mean}, std={std}")
    print(f"Hyperplane equation (in standardized space):")
    print(f"  {w[0]:.6f}*x + {w[1]:.6f}*y + {w[2]:.6f}*z + {b:.6f} = 0")
    print(f"Weights (w): {w}")
    print(f"Bias (b): {b}")
    print("=" * 60)

    return model


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting SVM Training")
    print("=" * 60 + "\n")

    # Train raw SVM
    model_raw = train_SVM()

    print("\n")

    # Train standardized SVM
    model_std = train_std_SVM()

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print("Models saved:")
    print("  - model/svm_raw.pkl (raw SVM)")
    print("  - model/svm_std.pkl (standardized SVM)")
    print("=" * 60 + "\n")