"""
1_drawSVM.py - SVM Test Set 3D Visualization
Loads SVM model, visualizes test set classification with hyperplane
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Ensure output directory exists
output_dir = os.path.join(os.path.dirname(__file__), 'picture')
os.makedirs(output_dir, exist_ok=True)


def load_model(model_path):
    """Load SVM model from pickle file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model, X):
    """Predict using SVM model (returns 0 or 1 labels)."""
    w = model['w']
    b = model['b']
    mean = model.get('mean', np.zeros(3))
    std = model.get('std', np.ones(3))

    # Standardize features if model has standardization params
    if 'mean' in model and 'std' in model:
        X_std = (X - mean) / std
    else:
        X_std = X

    # Compute decision function
    decision = np.dot(X_std, w) + b

    # Convert to 0/1 labels: decision >= 0 -> 1, decision < 0 -> 0
    # Note: In SVM, y=1 when decision > 0, y=-1 when decision < 0
    # Original labels: y=1 -> Label=1, y=-1 -> Label=0
    predictions = np.where(decision >= 0, 1, 0)

    return predictions


def compute_hyperplane_z(x, y, model):
    """
    Compute z coordinate on hyperplane for given x, y in original space.

    Hyperplane in standardized space: w[0]*x_std + w[1]*y_std + w[2]*z_std + b = 0
    Solve for z_std: z_std = (-w[0]*x_std - w[1]*y_std - b) / w[2]
    Convert back to original: z = z_std * std[2] + mean[2]
    """
    w = model['w']
    b = model['b']
    mean = model['mean']
    std = model['std']

    # Convert x, y to standardized space
    x_std = (x - mean[0]) / std[0]
    y_std = (y - mean[1]) / std[1]

    # Compute z_std
    z_std = (-w[0] * x_std - w[1] * y_std - b) / w[2]

    # Convert back to original space
    z = z_std * std[2] + mean[2]

    return z


def draw_svm_visualization():
    """Draw 3D visualization of SVM classification on test set."""
    # Load model
    model_path = 'model/svm_std.pkl'
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Weights: {model['w']}")
    print(f"Bias: {model['b']}")
    print(f"Mean: {model['mean']}")
    print(f"Std: {model['std']}")

    # Load test data
    test_data = pd.read_csv('data/test.csv')
    X_test = test_data[['Feature1', 'Feature2', 'Feature3']].values
    y_true = test_data['Label'].values

    # Make predictions
    y_pred = predict(model, X_test)

    # Find misclassified samples
    misclassified = y_true != y_pred
    correct = ~misclassified

    print(f"\nTest Set Classification Results:")
    print(f"Total samples: {len(y_true)}")
    print(f"Correct predictions: {np.sum(correct)}")
    print(f"Misclassified: {np.sum(misclassified)}")
    print(f"Accuracy: {100 * np.sum(correct) / len(y_true):.2f}%")

    # Extract coordinates
    X = X_test[:, 0]
    Y = X_test[:, 1]
    Z = X_test[:, 2]

    # Create 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot correctly classified samples
    # Label=0: blue (#1f77b4)
    mask_0_correct = correct & (y_true == 0)
    ax.scatter(X[mask_0_correct], Y[mask_0_correct], Z[mask_0_correct],
                c='#1f77b4', s=40, alpha=0.7, label='Label=0 (Correct)')

    # Label=1: red (#d62728)
    mask_1_correct = correct & (y_true == 1)
    ax.scatter(X[mask_1_correct], Y[mask_1_correct], Z[mask_1_correct],
                c='#d62728', s=40, alpha=0.7, label='Label=1 (Correct)')

    # Plot misclassified samples with black x markers
    if np.sum(misclassified) > 0:
        ax.scatter(X[misclassified], Y[misclassified], Z[misclassified],
                   c='#000000', s=80, marker='x', linewidths=2,
                   label='Misclassified')

    # Create mesh grid for hyperplane
    x_range = np.linspace(X.min() - 0.5, X.max() + 0.5, 50)
    y_range = np.linspace(Y.min() - 0.5, Y.max() + 0.5, 50)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)

    # Compute Z values for hyperplane
    Z_mesh = compute_hyperplane_z(X_mesh, Y_mesh, model)

    # Plot hyperplane
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh,
                    color='#808080', alpha=0.4,
                    label='Hyperplane')

    # Set axis labels
    ax.set_xlabel('Feature1')
    ax.set_ylabel('Feature2')
    ax.set_zlabel('Feature3')

    # Set title
    ax.set_title('SVM测试集分类结果三维可视化')

    # Add legend
    ax.legend(loc='upper left')

    # Show grid
    ax.grid(True)

    # Set view angle
    ax.view_init(elev=30, azim=-120)

    # Save figure
    output_path = os.path.join(output_dir, 'svm_test_3d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Show interactive window
    plt.show()


if __name__ == '__main__':
    draw_svm_visualization()