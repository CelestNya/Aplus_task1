"""
2_draw.py - PCA降维3D可视化
四分类任务：手动PCA实现 + MLP模型预测 + 3D可视化
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create picture directory
os.makedirs('picture', exist_ok=True)


class MLP4(nn.Module):
    """MLP4 model architecture for 4-class classification."""
    def __init__(self, input_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)


def manual_pca(X, n_components=3):
    """
    Manual PCA implementation using numpy.

    Args:
        X: Input data matrix (n_samples, n_features)
        n_components: Number of principal components

    Returns:
        X_pca: Transformed data (n_samples, n_components)
        top_eigenvectors: Selected eigenvectors (n_features, n_components)
    """
    # 1. 去中心化（减均值）
    X_centered = X - X.mean(axis=0)

    # 2. 协方差矩阵
    cov = np.cov(X_centered.T)

    # 3. 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 4. 取Top3特征向量（按特征值降序）
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    top_eigenvectors = eigenvectors[:, idx]

    # 5. 投影到主成分空间
    X_pca = X_centered @ top_eigenvectors

    return X_pca, top_eigenvectors


def load_model(model_path='model/mlp4.pth'):
    """Load trained MLP model and normalization parameters."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = MLP4(input_dim=16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    mean = checkpoint['mean']
    std = checkpoint['std']
    return model, mean, std


def load_test_data(data_path='data/gandou_test.csv'):
    """Load test data from CSV file."""
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col != 'Class']
    X = df[feature_cols].values
    y = df['Class'].values
    return X, y, feature_cols


def standardize_features(X, mean, std):
    """Standardize features using provided mean and std."""
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std


def predict(model, X_std):
    """Predict using the trained model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_std)
        outputs = model(X_tensor)
        preds = outputs.argmax(dim=1).numpy()
    return preds


def get_class_centroids(X_pca, y):
    """Calculate centroids for each class in PCA space."""
    centroids = []
    for cls in sorted(np.unique(y)):
        mask = y == cls
        centroid = X_pca[mask].mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def plot_3d_pca(X_pca, y_true, y_pred, centroids, feature_cols):
    """
    Create 3D PCA visualization with classification results.

    Args:
        X_pca: PCA-transformed data (n_samples, 3)
        y_true: True labels
        y_pred: Predicted labels
        centroids: Class centroids in PCA space
        feature_cols: Original feature column names
    """
    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for 4 classes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

    # Find misclassified samples
    correct_mask = y_true == y_pred
    wrong_mask = ~correct_mask

    # Plot correct classified samples for each class
    for cls in sorted(np.unique(y_true)):
        mask = correct_mask & (y_true == cls)
        if np.any(mask):
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                c=colors[cls], s=40, alpha=0.7,
                label=f'{class_names[cls]} (Correct: {mask.sum()})'
            )

    # Plot misclassified samples
    if np.any(wrong_mask):
        ax.scatter(
            X_pca[wrong_mask, 0], X_pca[wrong_mask, 1], X_pca[wrong_mask, 2],
            c='#000000', s=90, marker='x',
            label=f'Misclassified ({wrong_mask.sum()})'
        )

    # Plot class centroids (decision boundary approximation)
    for cls in sorted(np.unique(y_true)):
        ax.scatter(
            centroids[cls, 0], centroids[cls, 1], centroids[cls, 2],
            c=colors[cls], s=200, alpha=0.3,
            marker='o', edgecolors='black', linewidths=2
        )

    # Set labels and title
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.set_title('四分类测试集PCA降维三维可视化', fontsize=14, pad=10)

    # Set viewpoint
    ax.view_init(elev=25, azim=-130)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='upper left', fontsize=9)

    # Save figure
    save_path = 'picture/2_draw.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

    # Show interactive window
    plt.show()


def main():
    print("=" * 60)
    print("PCA 3D Visualization for 4-Class Classification")
    print("=" * 60)

    # 1. Load trained MLP model
    print("\n[1] Loading MLP model...")
    model, mean, std = load_model('model/mlp4.pth')
    print(f"Model loaded from: model/mlp4.pth")
    print(f"Normalization - mean shape: {mean.shape}, std shape: {std.shape}")

    # 2. Load test data
    print("\n[2] Loading test data...")
    X_raw, y_true, feature_cols = load_test_data('data/gandou_test.csv')
    print(f"Test data shape: {X_raw.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Classes: {sorted(np.unique(y_true))}")

    # 3. Standardize features for prediction
    print("\n[3] Standardizing features for prediction...")
    X_std = standardize_features(X_raw, mean, std)

    # 4. Predict using MLP model
    print("\n[4] Predicting with MLP model...")
    y_pred = predict(model, X_std)
    accuracy = (y_pred == y_true).mean()
    print(f"Prediction accuracy: {accuracy:.4f} ({(y_pred == y_true).sum()}/{len(y_true)})")
    print(f"Correct: {(y_pred == y_true).sum()}, Wrong: {(y_pred != y_true).sum()}")

    # 5. Manual PCA on raw (non-standardized) features
    print("\n[5] Performing manual PCA on raw features...")
    X_pca, eigenvectors = manual_pca(X_raw, n_components=3)
    print(f"PCA result shape: {X_pca.shape}")
    print(f"Eigenvectors shape: {eigenvectors.shape}")

    # 6. Calculate class centroids in PCA space
    print("\n[6] Calculating class centroids in PCA space...")
    centroids = get_class_centroids(X_pca, y_true)
    print(f"Centroids shape: {centroids.shape}")
    for cls in sorted(np.unique(y_true)):
        print(f"  Class {cls} centroid: [{centroids[cls, 0]:.4f}, {centroids[cls, 1]:.4f}, {centroids[cls, 2]:.4f}]")

    # 7. Create 3D visualization
    print("\n[7] Creating 3D PCA visualization...")
    plot_3d_pca(X_pca, y_true, y_pred, centroids, feature_cols)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()