"""
训练集3D散点图绘制脚本
读取data/train.csv，绘制Feature1/Feature2/Feature3的三维散点图，按Label分类着色
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
output_dir = os.path.join(os.path.dirname(__file__), 'picture')
os.makedirs(output_dir, exist_ok=True)

# 读取数据
data_path = os.path.join(os.path.dirname(__file__), 'data', 'train.csv')
df = pd.read_csv(data_path)

# 提取特征和标签
X = df['Feature1'].values
Y = df['Feature2'].values
Z = df['Feature3'].values
labels = df['Label'].values

# 创建3D图形
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 按标签分组绘制
label_0 = labels == 0
label_1 = labels == 1

# Label=0: 蓝色 (#1f77b4)
ax.scatter(X[label_0], Y[label_0], Z[label_0],
           c='#1f77b4', s=40, alpha=0.7, label='Label=0')

# Label=1: 红色 (#d62728)
ax.scatter(X[label_1], Y[label_1], Z[label_1],
           c='#d62728', s=40, alpha=0.7, label='Label=1')

# 设置坐标轴标签
ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
ax.set_zlabel('Feature3')

# 设置标题
ax.set_title('训练集数据三维分类散点图')

# 添加图例
ax.legend()

# 显示网格
ax.grid(True)

# 设置视角: elev=30, azim=-120
ax.view_init(elev=30, azim=-120)

# 保存图片
output_path = os.path.join(output_dir, 'train_3d_scatter.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {output_path}")

# 显示图形
plt.show()