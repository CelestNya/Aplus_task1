import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create model directory
os.makedirs('model', exist_ok=True)


class MLP4(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # 隐藏层1 64个神经元
            nn.ReLU(), # 激活函数
            nn.Dropout(0.2), # 随机失活20%神经元
            nn.Linear(64, 32), # 隐藏层2 32个神经元
            nn.ReLU(), # 激活函数
            nn.Dropout(0.2), # 随机失活20%神经元 
            nn.Linear(32, 4), # 输出层 4个神经元
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    """Load training and test data from CSV files."""
    train_df = pd.read_csv('data/gandou_train.csv')
    test_df = pd.read_csv('data/gandou_test.csv')

    feature_cols = [col for col in train_df.columns if col != 'Class']

    X_train = train_df[feature_cols].values
    y_train = train_df['Class'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['Class'].values

    return X_train, y_train, X_test, y_test


def standardize_features(X_train, X_test):
    """Standardize features using training set mean and std."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_test_std, mean, std


def train_model(X_train, y_train, X_val, y_val, epochs=300, patience=30):
    """Train the MLP4 model with early stopping and learning rate scheduling."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)

    # Create model
    model = MLP4(input_dim=X_train.shape[1]).to(device)
    print(f"Model architecture:\n{model}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Training variables
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Print loss every epoch
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val_tensor).float().mean().item()

        # Print validation accuracy every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Val Accuracy: {val_acc:.4f}")

        # Update learning rate
        scheduler.step(val_acc)

        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


def save_model(model, mean, std, filepath='model/mlp4.pth'):
    """Save model state dict along with normalization parameters."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std
    }, filepath)
    print(f"Model saved to {filepath}")


def main():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")

    print("\nStandardizing features...")
    X_train_std, X_test_std, mean, std = standardize_features(X_train, X_test)

    print("\nTraining MLP4...")
    model, best_val_acc = train_model(X_train_std, y_train, X_test_std, y_test)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    print("\nSaving model...")
    save_model(model, mean, std)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()