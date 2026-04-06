# Aplus ML分类任务设计规格

## 1. 项目概述

- **项目名称**：Aplus ML分类任务
- **类型**：机器学习双任务并行项目（二分类SVM + 四分类神经网络）
- **核心目标**：实现二分类SVM（准确率≥90%）和四分类NN（准确率≥75%），全程自循环调参达标
- **约束**：仅使用torch、pandas、numpy、matplotlib；禁止sklearn、joblib等第三方库

---

## 2. 任务1：二分类SVM（1_*.py系列）

### 2.1 数据
- `data/train.csv`：Feature1, Feature2, Feature3, Label（Label∈{0,1}）
- `data/test.csv`：同上格式

### 2.2 文件清单与职责

| 文件 | 职责 |
|------|------|
| `1_draw.py` | 读取train.csv，3D散点图可视化（Label=0蓝，Label=1红），elev=30, azim=-120，高清保存+交互窗口 |
| `1_train.py` | `train_SVM()`：numpy实现SGD型线性SVM（铰链损失），保存`./model/svm_raw.pkl`；`train_std_SVM()`：标准化特征后训练，保存`./model/svm_std.pkl`；打印超平面方程 |
| `1_test.py` | 加载两个SVM模型，对test.csv预测；若准确率<90%，触发自循环（调参→重训练）|
| `1_drawSVM.py` | 加载SVM模型，预测test.csv；绘制3D散点（正确=蓝/红，错误=黑s=80）+ 灰色半透明超平面，elev=30, azim=-120，高清保存+交互窗口 |

### 2.3 SVM算法实现细节
- **算法**：随机梯度下降（SGD）求解铰链损失最小化
- **目标函数**：`min_{w,b} (1/2)||w||^2 + C * sum(max(0, 1 - y_i*(w*x_i + b)))`
- **参数搜索空间**：
  - C ∈ {0.1, 1.0, 10.0, 100.0}
  - learning_rate ∈ {0.001, 0.01, 0.1}
  - epochs ∈ {500, 1000, 2000, 5000}
- **自循环策略**：测试准确率<90% → 调整C/学习率/迭代次数 → 重新训练 → 重新测试，至达标

### 2.4 绘图规范
- 中文字体支持：使用`SimHei`或其他可用中文字体
- 颜色：Label0=#1f77b4（蓝），Label1=#d62728（红），错误=#000000（黑）
- 超平面：color='#808080'，alpha=0.4
- 保存：dpi=300，bbox_inches='tight'

---

## 3. 任务2：四分类神经网络（2_*.py系列）

### 3.1 数据
- `data/gandou.csv`：16个特征 + Class（Class∈{0,1,2,3}）
- 划分：8:2 → `gandou_train.csv`、`gandou_test.csv`

### 3.2 文件清单与职责

| 文件 | 职责 |
|------|------|
| `2_divide.py` | 读取gandou.csv，8:2划分，保证类别分布均衡，保存至data/ |
| `2_train.py` | 基于torch实现3层全连接NN（16→64→32→4），ReLU+Dropout，Adam优化器，保存`./model/mlp4.pth`；自循环调参至测试准确率≥75% |
| `2_test.py` | 加载MLP模型，对gandou_test.csv预测；若准确率<75%，触发自循环（调网络/学习率等） |
| `2_draw.py` | 手动PCA降维（16D→3D），绘制3D散点（四色），错误点黑s=90；绘制决策边界（半透明alpha=0.3），elev=25, azim=-130，高清保存+交互窗口 |

### 3.3 神经网络架构
```
Input(16) → Linear(16, 64) → ReLU → Dropout(0.2) → Linear(64, 32) → ReLU → Dropout(0.2) → Linear(32, 4)
```
- **优化器**：Adam（lr初始0.001）
- **损失函数**：CrossEntropyLoss
- **正则化**：Dropout(0.2) + 权重衰减（1e-4）
- **自循环策略**：准确率<75% → 调整网络宽度（64/32→128/64等）、学习率、BatchSize、正则强度，至达标

### 3.4 PCA手动实现
1. 去中心化（减均值）
2. 计算协方差矩阵（16×16）
3. 特征分解，取Top3特征向量
4. 投影测试数据到Top3主成分

### 3.5 绘图规范
- 四分类颜色：#1f77b4（蓝）、#ff7f0e（橙）、#2ca02c（绿）、#9467bd（紫）
- 错误点：#000000，s=90
- 决策边界：alpha=0.3
- 坐标轴：PCA1, PCA2, PCA3
- 保存：dpi=300，bbox_inches='tight'

---

## 4. 全局机制

### 4.1 自循环自审
- 1_test.py：准确率<90% → 打印警告 → 调用1_train.py重新训练（调整超参数）
- 2_test.py：准确率<75% → 打印警告 → 调用2_train.py重新训练（调整网络/超参数）
- 循环上限：10次，防止无限循环

### 4.2 目录结构
```
./data/          - train.csv, test.csv, gandou.csv, gandou_train.csv, gandou_test.csv
./model/         - svm_raw.pkl, svm_std.pkl, mlp4.pth
./picture/       - 所有可视化图片（1_draw.png, 1_drawSVM.png, 2_draw.png）
```

### 4.3 运行方式
所有脚本通过`uv run python <script>.py`执行

---

## 5. 验收标准
- [ ] 1_draw.py：训练集3D散点图，弹出交互窗口，高清保存
- [ ] 1_train.py：两个SVM模型均保存，打印超平面方程
- [ ] 1_test.py：测试准确率≥90%，否则自循环调参
- [ ] 1_drawSVM.py：测试集分类可视化，错误点黑色突出，超平面可见，弹出交互窗口
- [ ] 2_divide.py：gandou数据8:2划分，类别分布均衡
- [ ] 2_train.py：MLP模型保存，训练过程打印loss
- [ ] 2_test.py：测试准确率≥75%，否则自循环调参
- [ ] 2_draw.py：PCA降维3D可视化，决策边界可见，弹出交互窗口
- [ ] 所有图片dpi=300，中文字体正常显示
