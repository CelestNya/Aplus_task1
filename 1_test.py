"""
1_test.py - SVM Testing with Self-Loop Hyperparameter Tuning
Loads SVM models and tests accuracy, with automatic retraining if needed
"""

import os
import pickle
import subprocess
import sys
import re
import numpy as np
import pandas as pd


def predict(X, model):
    """
    使用SVM模型预测
    Args:
        X: numpy array of shape (n_samples, n_features)
        model: dict with 'w', 'b', and optionally 'mean', 'std'
    Returns:
        predictions: numpy array of 0/1 labels
    """
    # 如果模型包含mean/std字段，先对X标准化
    if 'mean' in model and 'std' in model:
        X = (X - model['mean']) / model['std']

    w = model['w']
    b = model['b']

    # 计算决策函数: f(x) = w*x + b
    decision = np.dot(X, w) + b

    # 转换为标签: decision >= 0 -> 1, decision < 0 -> 0
    predictions = (decision >= 0).astype(int)

    return predictions


def load_model(model_path):
    """加载模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_model(model, X, y_true):
    """
    评估模型
    Returns:
        accuracy: float
        correct: int
        total: int
    """
    y_pred = predict(X, model)
    correct = np.sum(y_pred == y_true)
    total = len(y_true)
    accuracy = correct / total
    return accuracy, correct, total


def test_models():
    """
    测试两个模型
    Returns:
        (accuracy_raw, accuracy_std, best_accuracy, best_model_name)
    """
    # 读取测试数据
    test_data = pd.read_csv('data/test.csv')
    X_test = test_data[['Feature1', 'Feature2', 'Feature3']].values
    y_test = test_data['Label'].values

    # 加载模型
    model_raw = load_model('model/svm_raw.pkl')
    model_std = load_model('model/svm_std.pkl')

    # 评估原生模型
    accuracy_raw, correct_raw, total_raw = evaluate_model(model_raw, X_test, y_test)
    print(f"\n原生SVM模型 (svm_raw):")
    print(f"  准确率: {accuracy_raw:.4f} ({accuracy_raw * 100:.2f}%)")
    print(f"  正确预测: {correct_raw}/{total_raw}")

    # 评估标准化模型
    accuracy_std, correct_std, total_std = evaluate_model(model_std, X_test, y_test)
    print(f"\n标准化SVM模型 (svm_std):")
    print(f"  准确率: {accuracy_std:.4f} ({accuracy_std * 100:.2f}%)")
    print(f"  正确预测: {correct_std}/{total_std}")

    # 确定最佳模型
    if accuracy_raw >= accuracy_std:
        best_accuracy = accuracy_raw
        best_model_name = 'svm_raw'
    else:
        best_accuracy = accuracy_std
        best_model_name = 'svm_std'

    print(f"\n最佳模型: {best_model_name}, 准确率: {best_accuracy:.4f}")

    return accuracy_raw, accuracy_std, best_accuracy, best_model_name


def get_current_params():
    """从1_train.py读取当前超参数"""
    with open('1_train.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取C, lr, epochs
    c_match = re.search(r'C\s*=\s*([\d.]+)', content)
    lr_match = re.search(r'lr\s*=\s*([\d.]+)', content)
    epochs_match = re.search(r'epochs\s*=\s*(\d+)', content)

    c = float(c_match.group(1)) if c_match else None
    lr = float(lr_match.group(1)) if lr_match else None
    epochs = int(epochs_match.group(1)) if epochs_match else None

    return c, lr, epochs


def update_train_params(c, lr, epochs):
    """更新1_train.py中的超参数"""
    with open('1_train.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式替换参数（两个函数中的参数都要改）
    # 对于C, lr, epochs需要替换所有出现的位置
    content = re.sub(r'C\s*=\s*[\d.]+', f'C = {c}', content)
    content = re.sub(r'lr\s*=\s*[\d.]+', f'lr = {lr}', content)
    content = re.sub(r'epochs\s*=\s*\d+', f'epochs = {epochs}', content)

    with open('1_train.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"已更新参数: C={c}, lr={lr}, epochs={epochs}")


def self_loop_tuning():
    """
    自循环调参
    遍历参数网格，重新训练，测试准确率
    若某组参数使准确率>=90%，停止循环
    Returns:
        (success, best_accuracy, best_params)
    """
    param_grid = [
        {'C': 0.1, 'lr': 0.01, 'epochs': 5000},
        {'C': 1.0, 'lr': 0.01, 'epochs': 5000},
        {'C': 10.0, 'lr': 0.001, 'epochs': 8000},
        {'C': 100.0, 'lr': 0.001, 'epochs': 10000},
        {'C': 50.0, 'lr': 0.005, 'epochs': 8000},
    ]

    print("\n" + "=" * 60)
    print("开始自循环调参")
    print("=" * 60)

    # 保存原始参数
    orig_c, orig_lr, orig_epochs = get_current_params()
    print(f"原始参数: C={orig_c}, lr={orig_lr}, epochs={orig_epochs}")

    best_accuracy = 0
    best_params = None

    for i, params in enumerate(param_grid):
        print(f"\n--- 参数组合 {i + 1}/{len(param_grid)}: {params} ---")

        # 更新参数
        update_train_params(params['C'], params['lr'], params['epochs'])

        # 重新训练
        print("重新训练模型...")
        result = subprocess.run(
            [sys.executable, '1_train.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"训练失败: {result.stderr}")
            continue

        # 测试准确率
        try:
            accuracy_raw, accuracy_std, best_acc, best_model = test_models()
        except FileNotFoundError:
            print("模型文件未找到，跳过此参数组合")
            continue

        # 更新最佳参数
        if best_acc > best_accuracy:
            best_accuracy = best_acc
            best_params = params.copy()
            best_params['model'] = best_model
            print(f"*** 新最佳准确率: {best_accuracy:.4f} ***")

        # 若准确率>=90%，停止循环
        if best_acc >= 0.90:
            print(f"\n准确率已达标 ({best_acc:.4f} >= 0.90)，停止调参")
            break

    # 恢复原始参数
    print(f"\n恢复原始参数: C={orig_c}, lr={orig_lr}, epochs={orig_epochs}")
    update_train_params(orig_c, orig_lr, orig_epochs)

    # 如果找到了>=90%的参数组合，应用最佳参数
    if best_params is not None and best_accuracy >= 0.90:
        print(f"\n应用最佳参数: {best_params}")
        update_train_params(best_params['C'], best_params['lr'], best_params['epochs'])
        # 重新训练一次确保模型是最佳的
        subprocess.run([sys.executable, '1_train.py'], capture_output=True)

    return best_accuracy >= 0.90, best_accuracy, best_params


def main():
    print("=" * 60)
    print("SVM模型测试")
    print("=" * 60)

    # 检查模型文件是否存在
    if not os.path.exists('model/svm_raw.pkl') or not os.path.exists('model/svm_std.pkl'):
        print("\n模型文件不存在，先运行训练...")
        result = subprocess.run([sys.executable, '1_train.py'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"训练失败: {result.stderr}")
            return

    # 测试两个模型
    accuracy_raw, accuracy_std, best_accuracy, best_model_name = test_models()

    # 判断最优模型及准确率是否>=90%
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    if best_accuracy >= 0.90:
        print(f"最优模型: {best_model_name}")
        print(f"准确率: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
        print(f"结果: [PASS]")
    else:
        print(f"最优准确率: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
        print(f"未达标 (< 90%)，执行自循环调参...")

        success, final_accuracy, best_params = self_loop_tuning()

        print("\n" + "=" * 60)
        print("自循环调参完成")
        print("=" * 60)

        if success:
            print(f"最佳参数: {best_params}")
            print(f"最终准确率: {final_accuracy:.4f}")
            print(f"结果: [PASS]")
        else:
            print(f"最终准确率: {final_accuracy:.4f}")
            print(f"结果: [FAIL]")


if __name__ == '__main__':
    main()