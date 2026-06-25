# GS_PINO

基于 U-FNO 神经算子的固定边界 Grad-Shafranov 方程求解器。

## 快速开始

```bash
# 1. 生成数据集
python -m gs_pino.generate_dataset --out data/gs_demo.npz --n-samples 20 --nr 65 --nz 65

# 2. 训练模型
python -m gs_pino.train --data data/gs_demo.npz --epochs 10 --batch-size 4 --width 16 --modes1 8 --modes2 8 --layers 2 --pde-weight 0.01 --output-dir outputs/demo

# 3. 评估模型
python -m gs_pino.evaluate --data data/gs_demo.npz --checkpoint outputs/demo/best.pt --output-dir outputs/demo_eval --max-plots 3
```

## 安装

```bash
git clone <repo-url> && cd GS_PINO
pip install -e .
```

**外部求解器**（可选，用于生成真实训练数据）：[GS_solver](https://github.com/994148196/GS_solver) 需克隆至同级目录。

## 命令行参数

### generate_dataset

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--out` | `data/gs_fixed_boundary.npz` | 输出路径 |
| `--n-samples` | 64 | 样本数 |
| `--nr`, `--nz` | 64, 64 | 网格点数 |
| `--rtol` | 1e-5 | 求解器收敛容差 |
| `--seed` | 42 | 随机种子 |

### train

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/gs_fixed_boundary.npz` | 数据集路径 |
| `--output-dir` | `outputs/run` | 输出目录 |
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 4 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--width` | 32 | 模型宽度 |
| `--modes1/2` | 16/16 | 傅里叶模态数 |
| `--layers` | 4 | U-FNO 块数 |
| `--pde-weight` | 0.01 | PDE 残差损失权重 |
| `--bc-weight` | 0.05 | 边界损失权重 |
| `--ip-weight` | 0.0 | Ip 约束损失权重 |

### evaluate

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/gs_fixed_boundary.npz` | 数据集路径 |
| `--checkpoint` | (必填) | 模型权重文件 |
| `--output-dir` | `outputs/eval` | 输出目录 |
| `--max-plots` | 12 | 可视化样本数 |

## 目录结构

```
GS_PINO/
├── src/gs_pino/
│   ├── data.py           # 数据集加载与输入通道构建
│   ├── evaluate.py       # 评估与可视化
│   ├── generate_dataset.py  # 数据集生成
│   ├── geometry.py       # LCFS 几何工具
│   ├── losses.py         # 损失函数（含 GS 残差）
│   ├── models.py         # U-FNO 模型定义
│   ├── solvers.py        # GS 求解器适配器
│   └── train.py          # 训练脚本
├── data/                 # 数据集目录
├── outputs/              # 训练/评估输出
├── README.md
├── TECH_DOC.md           # 详细技术文档
└── pyproject.toml
```

## 更多

详细技术文档（模型架构、变量形状、损失函数推导等）请见 [TECH_DOC.md](TECH_DOC.md)。
