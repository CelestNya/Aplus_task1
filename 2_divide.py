"""
2_divide.py - Gandou dataset 8:2 stratified split
"""
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Load data
data_path = 'data/gandou.csv'
df = pd.read_csv(data_path)

print("=" * 60)
print("Gandou Dataset Split Report (8:2 stratified)")
print("=" * 60)

# Total samples
total_samples = len(df)
print(f"\nTotal samples: {total_samples}")

# Class distribution in original data
print("\nOriginal class distribution:")
class_counts = df['Class'].value_counts().sort_index()
class_percentages = (class_counts / total_samples * 100).round(2)
for cls in sorted(df['Class'].unique()):
    count = class_counts[cls]
    pct = class_percentages[cls]
    print(f"  Class {cls}: {count} samples ({pct}%)")

# Stratified split 8:2 - manual implementation
train_indices = []
test_indices = []

for cls in sorted(df['Class'].unique()):
    cls_indices = df[df['Class'] == cls].index.tolist()
    np.random.shuffle(cls_indices)
    split_idx = int(len(cls_indices) * 0.8)
    train_indices.extend(cls_indices[:split_idx])
    test_indices.extend(cls_indices[split_idx:])

# Create train and test dataframes
train_df = df.loc[train_indices].reset_index(drop=True)
test_df = df.loc[test_indices].reset_index(drop=True)

# Save to CSV
train_path = 'data/gandou_train.csv'
test_path = 'data/gandou_test.csv'
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# Training set statistics
train_samples = len(train_df)
print(f"\nTraining set: {train_samples} samples ({(train_samples/total_samples*100):.1f}%)")
print("Training set class distribution:")
train_class_counts = train_df['Class'].value_counts().sort_index()
for cls in sorted(train_df['Class'].unique()):
    count = train_class_counts[cls]
    pct = (count / train_samples * 100) if train_samples > 0 else 0
    print(f"  Class {cls}: {count} samples ({pct:.2f}%)")

# Test set statistics
test_samples = len(test_df)
print(f"\nTest set: {test_samples} samples ({(test_samples/total_samples*100):.1f}%)")
print("Test set class distribution:")
test_class_counts = test_df['Class'].value_counts().sort_index()
for cls in sorted(test_df['Class'].unique()):
    count = test_class_counts[cls]
    pct = (count / test_samples * 100) if test_samples > 0 else 0
    print(f"  Class {cls}: {count} samples ({pct:.2f}%)")

print("\n" + "=" * 60)
print(f"Files saved: {train_path}, {test_path}")
print("=" * 60)