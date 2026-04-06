# Aplus ML Classification Task1

A machine learning classification project implementing two parallel tasks from scratch — **binary classification via SVM** and **four-class classification via neural network (MLP)**.

## Project Overview

| Task | Algorithm | Dataset | Accuracy Target |
|------|-----------|---------|----------------|
| Binary Classification | SVM (SGD, hinge loss) | `data/train.csv`, `data/test.csv` | ≥ 90% |
| Four-Class Classification | MLP (3-layer FCNN) | `data/gandou.csv` | ≥ 75% |

## Tech Stack

- **PyTorch** — neural network
- **NumPy** — SVM implementation, PCA
- **Pandas** — data loading
- **Matplotlib** — 3D visualization

## Project Structure

```
data/           — train.csv, test.csv, gandou.csv, gandou_train.csv, gandou_test.csv
model/          — svm_raw.pkl, svm_std.pkl, mlp4.pth
picture/        — 3D visualization outputs
1_draw.py      — Task1: 3D scatter of training data
1_train.py     — Task1: train raw + standardized SVM
1_test.py      — Task1: evaluate SVM (self-tuning loop)
1_drawSVM.py   — Task1: visualize SVM decision boundary
2_divide.py    — Task2: split gandou.csv 8:2
2_train.py     — Task2: train 3-layer MLP
2_test.py      — Task2: evaluate MLP (self-tuning loop)
2_draw.py      — Task2: PCA → 3D + decision boundary
```

## Running

```bash
uv run python 1_draw.py
uv run python 1_train.py
uv run python 1_test.py
uv run python 1_drawSVM.py
uv run python 2_divide.py
uv run python 2_train.py
uv run python 2_test.py
uv run python 2_draw.py
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
