## 项目基础约束
1. 项目由`uv init`构建，**仅允许使用torch、pandas、numpy、matplotlib**，禁止引入scikit-learn、joblib及任何第三方库
2. 固定目录：根目录含`data/`（内置train.csv、test.csv、gandou.csv），自动创建`model/`存模型、自动适配图片保存路径
3. 固定数据表头
    - train.csv/test.csv：`Feature1,Feature2,Feature3,Label`
    - gandou.csv：`Area,Perimeter,MajorAxisLength,MinorAxisLength,AspectRation,Eccentricity,ConvexArea,EquivDiameter,Extent,Solidity,roundness,Compactness,ShapeFactor1,ShapeFactor2,ShapeFactor3,ShapeFactor4,Class`
4. 核心机制：**全程自动循环自审优化**，准确率未达标则自主调整模型结构、训练参数、优化器、迭代次数等，直至满足硬性指标
5. 代码无冗余、无报错，路径均为相对路径，运行后直接输出结果，所有可视化图片**高清保存+弹出交互式3D窗口**

## 任务硬性指标
- 任务1（二分类SVM）：测试集准确率≥90%
- 任务2（四分类）：测试集准确率≥75%

## 任务分工（双Subagent并行执行）
### Subagent 1：负责`1_*.py`系列（二分类SVM任务）
需生成文件：`1_draw.py、1_train.py、1_test.py、1_drawSVM.py`

#### 1_draw.py
1. 读取`data/train.csv`，提取Feature1、Feature2、Feature3作为三维坐标，Label为分类标签
2. **绘图样式规范**
    - 3D散点：Label=0用**蓝色(#1f77b4)**，Label=1用**红色(#d62728)**，点大小`s=40`，透明度`alpha=0.7`
    - 坐标轴：X轴命名Feature1、Y轴Feature2、Z轴Feature3，显示坐标轴网格
    - 标题：`训练集数据三维分类散点图`，添加清晰图例标注分类
    - 三维视角：固定`elev=30, azim=-120`，保证可视化效果最佳
3. 输出：高清保存图片（dpi=300，bbox_inches='tight'），**调用plt.show()弹出可旋转、缩放的交互式3D窗口**

#### 1_train.py
1. 实现`train_SVM()`：原生实现线性SVM模型，基于train.csv训练，模型保存至`./model/`，打印完整超平面方程
2. 实现`train_std_SVM()`：对特征做标准化处理后训练SVM，模型保存至`./model/`，打印超平面方程
3. 训练参数、优化方式、迭代轮数等所有技术细节自主确定，自动调参直至测试准确率达标

#### 1_test.py
1. 加载两个SVM模型，对`test.csv`完成预测
2. 打印测试准确率、正确预测数、总样本数等核心结果概况
3. 自动校验准确率是否≥90%，未达标则反馈至训练环节循环调参、重新训练

#### 1_drawSVM.py
1. 加载SVM模型，对测试集完成预测并标记分类错误样本
2. **绘图样式规范**
    - 3D散点：正确分类样本Label0蓝、Label1红（s=40，alpha=0.7）；分类错误样本用**纯黑色(#000000)**，点大小`s=80`高亮突出
    - 超平面：绘制灰色半透明平面（`color='#808080'`，alpha=0.4），清晰展示分类边界
    - 图例：标注类别0、类别1、分类超平面、分类错误点，坐标轴与视角同1_draw.py
    - 标题：`SVM测试集分类结果三维可视化`
3. 输出：高清保存图片（dpi=300）到picture文件夹，**调用plt.show()弹出交互式3D窗口**

---

### Subagent 2：负责`2_*.py`系列（四分类任务）
需生成文件：`2_divide.py、2_train.py、2_test.py、2_draw.py`

#### 2_divide.py
1. 读取`gandou.csv`，按8:2固定比例划分为`gandou_train.csv`和`gandou_test.csv`，保存至data目录
2. 划分逻辑可复现，保证训练集与测试集类别分布均衡，细节自主实现

#### 2_train.py
1. 针对gandou四分类任务，自主设计模型架构（神经网络结构自主确定）
2. 基于`gandou_train.csv`完成训练，模型保存至`./model/`
3. 自主调整网络层数、神经元数量、学习率、优化器等参数，自动优化直至测试准确率≥75%

#### 2_test.py
1. 加载训练完成的四分类模型，对`gandou_test.csv`完成预测
2. 打印测试准确率、各类别预测情况等结果概况
3. 自动校验准确率是否≥75%，未达标则反馈至训练环节循环调参、重构模型

#### 2_draw.py（新增，PCA三维可视化）
1. 数据处理：对gandou_test.csv的16维高维特征，**手动实现PCA降维至3维空间**
2. **绘图样式规范**
    - 3D散点：四分类标签分别使用**蓝(#1f77b4)、橙(#ff7f0e)、绿(#2ca02c)、紫(#9467bd)**四种区分色，点大小`s=40`，alpha=0.7
    - 错误样本：分类错误点用**纯黑色(#000000)**，点大小`s=90`高亮突出
    - 决策边界：基于训练好的四分类模型，绘制对应分类超平面/决策边界，半透明展示（alpha=0.3）
    - 坐标轴：X轴PCA1、Y轴PCA2、Z轴PCA3，显示网格，添加清晰图例标注4个类别+错误点
    - 标题：`四分类测试集PCA降维三维可视化`，视角`elev=25, azim=-130`
3. 输出：高清保存图片（dpi=300，bbox_inches='tight'），**调用plt.show()弹出可交互式3D窗口**

## 全局执行规则
1. 所有算法实现、参数配置、训练策略等技术细节由你自主确定，无需固定写法
2. 严格执行**自动自审循环**：准确率不达标→自动调整参数/结构→重新训练测试，直至满足指标
3. PCA、SVM、模型训练、可视化均仅使用torch/numpy/matplotlib原生实现
4. 一次性生成全部8个代码文件，逻辑闭环，运行后直接完成保存图片+弹出交互式3D图像的全部操作
5. 自行编写测试脚本验证成果，任务完成后删除多余文件
6. **项目由uv环境管理**请使用uv run命令运行
7. 绘图要有中文支持，导入中文字体，并且仔细审核绘图传入的参数，避免点阵在一个平面上
8. 各类输出文件的文件名（如模型文件）要与任务相关（避免分不清其输出来源）
9. 无需过问我更多问题，均选取你的推荐项目,提示词保存在./res/prompt.md中。

下面开始头脑风暴，并行调用多Subagent精确完成上述任务