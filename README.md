# GS_PINO: Grad-Shafranov Physics-Informed Neural Operator

GS_PINO是一个基于物理信息神经网络的Grad-Shafranov方程代理模型，用于预测托卡马克等离子体平衡。项目支持固定边界和自由边界两种模式，采用PlaNet-equil架构实现高效的物理约束训练。

## 目录结构

```
GS_PINO/
├── src/
│   └── gs_pino/
│       ├── __init__.py
│       ├── models.py              # PlaNetCore神经网络架构
│       ├── geometry.py            # 几何工具函数
│       ├── data.py                # 固定边界数据集
│       ├── data_freegs.py         # 自由边界数据集（含网格插值）
│       ├── losses.py              # 固定边界损失函数
│       ├── losses_freegs.py       # 自由边界损失函数（含GS算子）
│       ├── train.py               # 固定边界训练脚本
│       ├── train_freegs.py        # 自由边界训练脚本（GPU强制）
│       ├── evaluate.py            # 固定边界评估脚本
│       ├── evaluate_freegs.py     # 自由边界评估脚本
│       ├── generate_dataset.py    # 固定边界数据集生成
│       ├── generate_freegs_dataset.py  # 自由边界数据集生成
│       ├── inference_freegs.py    # 自由边界推理脚本
│       └── visualize_freegs.py    # 自由边界可视化脚本
├── data/                          # 数据集目录
├── outputs/                       # 训练输出目录
├── TECH_DOC.md                    # 技术文档
├── TECH_DOC.pdf                   # 技术文档PDF
└── pyproject.toml                 # 项目配置
```

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0（带CUDA支持）
- freegs（用于自由边界数据集生成）
- numpy, scipy, matplotlib, tqdm, joblib

推荐使用torch5060 conda环境：
```bash
conda activate torch5060
```

## 快速开始

### 1. 生成数据集（自由边界）

```bash
python -m src.gs_pino.generate_freegs_dataset \
    --out data/freegs_rhs_500.npz \
    --n-samples 500 \
    --nx 65 \
    --ny 65 \
    --seed 42 \
    --n-jobs -1
```

参数说明：
- `--out`: 输出文件路径
- `--n-samples`: 生成样本数量（建议500-1000）
- `--nx/--ny`: 网格点数（freegs要求2^n+1，如65=2^6+1）
- `--seed`: 随机种子
- `--n-jobs`: 并行工作数（-1为全部CPU核心）

### 2. 训练模型（自由边界）

```bash
python -m src.gs_pino.train_freegs \
    --data data/freegs_rhs_500.npz \
    --output-dir outputs/freegs_rhs_planet \
    --epochs 500 \
    --batch-size 8 \
    --lr 5e-4 \
    --hidden-dim 128 \
    --scale-mse 1.0 \
    --scale-pde 0.1 \
    --clip-grad 1.0 \
    --patience 50 \
    --min-epochs 200 \
    --warmup-epochs 10 \
    --seed 42
```

参数说明：
- `--data`: 数据集路径
- `--output-dir`: 输出目录（保存模型和历史）
- `--epochs`: 训练轮数
- `--batch-size`: 批量大小
- `--lr`: 学习率
- `--hidden-dim`: 网络隐藏层维度
- `--scale-mse`: MSE损失权重
- `--scale-pde`: PDE损失权重
- `--clip-grad`: 梯度裁剪范数
- `--patience`: 早期停止耐心值
- `--min-epochs`: 最小训练轮数
- `--warmup-epochs`: 学习率预热轮数

### 3. 评估模型

```bash
python -m src.gs_pino.evaluate_freegs \
    --data data/freegs_rhs_500.npz \
    --checkpoint outputs/freegs_rhs_planet/best.pt
```

## 架构设计

### PlaNetCore神经网络

采用Trunk-Branch-Decoder架构：

```
输入层:
  ├── Branch (标量参数) → 线圈电流 + 等离子体参数 → 全连接层 → 特征向量
  └── Trunk (空间坐标)  → R, Z坐标网格 → 卷积层 → 特征图
            ↓
Decoder: Branch特征向量 ⊙ Trunk特征图 → 卷积层 → psi_total输出
```

#### Trunk网络
处理空间坐标(R,Z)，提取空间特征：
- 输入：R网格 (64×64), Z网格 (64×64)
- 结构：层归一化 + 多层卷积 + ReLU激活
- 输出：空间特征图 (64×64×hidden_dim)

#### Branch网络
处理标量参数，提取参数特征：
- 输入：线圈电流(4) + 等离子体参数(5) = 9维
- 结构：多层全连接层 + ReLU激活
- 输出：参数特征向量 (hidden_dim)

#### Decoder网络
融合空间特征和参数特征：
- 操作：参数特征向量 ⊙ 空间特征图（逐元素乘法）
- 结构：多层卷积 + 最后一层线性层
- 输出：psi_total (64×64)

### 损失函数

采用混合损失函数：

```
Loss = λ_mse × MSE(pred, target) + λ_pde × PDE_loss(pred_plasma, rhs)
```

#### MSE损失
```python
MSE = mean((pred - target)²)
```
计算整个计算域上的均方误差。

#### PDE损失（Grad-Shafranov算子）
基于卷积核离散化Grad-Shafranov方程：

```python
∇²ψ_plasma - (1/R)∂ψ_plasma/∂R = -μ₀ R J(ψ_plasma)
```

实现步骤：
1. 计算Laplace算子卷积核
2. 计算径向导数卷积核
3. 通过卷积计算算子左边
4. 与真实RHS比较（MSE）

**关键设计**：网络预测psi_total，但PDE损失仅应用于psi_plasma = psi_total - psi_coils，确保物理约束仅作用于等离子体贡献。

### 课程学习策略

训练分为三个阶段：

| 阶段 | Epoch范围 | PDE权重 | 说明 |
|------|-----------|---------|------|
| Phase 1 | 1-100 | 0.0 | 纯MSE训练，建立基本通量分布 |
| Phase 2 | 101-150 | 0→1.0 | PDE损失线性递增 |
| Phase 3 | 151+ | 1.0 | 完整损失训练 |

这种策略避免了初始阶段PDE损失过大导致的训练不稳定。

## 数据格式

### 输入参数

| 参数 | 范围 | 说明 |
|------|------|------|
| Ip | 1.5e5 - 2.5e5 A | 等离子体电流 |
| paxis | 800 - 1500 Pa | 轴上压强 |
| alpha_m | 1.0 - 2.0 | 压强剖面形状参数 |
| alpha_n | 1.5 - 2.5 | 压强剖面形状参数 |
| fvac | 1.8 - 2.2 | 真空磁通量函数 |
| P1L, P1U, P2L, P2U | - | PF线圈电流（由freegs自动求解） |

### 输出数据

| 字段 | 形状 | 说明 |
|------|------|------|
| R | (64, 64) | R坐标网格 |
| Z | (64, 64) | Z坐标网格 |
| psi_total | (64, 64) | 总极向通量 |
| psi_plasma | (64, 64) | 等离子体通量 |
| psi_coils | (64, 64) | 线圈通量 |
| rhs | (62, 62) | GS方程右边项 |
| greens | (4, 64, 64) | 格林函数 |
| coil_currents | (4,) | 线圈电流 |
| mask | (64, 64) | 等离子体区域掩码 |
| params | (5,) | 等离子体参数 |

## 网格处理

### freegs网格要求

freegs求解器要求网格大小为2^n + 1（如65=2^6+1），这是由于其内部使用FFT求解器。

### 训练网格

训练时将65×65网格插值到64×64：
- 使用`scipy.interpolate.RegularGridInterpolator`
- 插值方法：quintic（五次多项式）
- 优势：64是2的幂，便于高效FFT运算

## GPU加速

训练强制要求CUDA：

```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available! Please use a GPU-enabled environment.")
```

### 优化策略

1. **混合精度训练**：通过`--amp`启用自动混合精度
2. **梯度累积**：通过`--accum-steps`实现更大批量效果
3. **梯度裁剪**：防止梯度爆炸
4. **Cosine退火**：学习率逐渐衰减

## 训练结果

### 典型训练曲线

训练过程中监控以下指标：
- MSE损失（数据拟合）
- PDE损失（物理约束）
- 总损失（加权和）
- 学习率变化

### 评估指标

- **MSE**：均方误差
- **相对误差**：||pred - target|| / ||target||

## 使用示例

### 推理

```python
import torch
from src.gs_pino.models import PlaNetCore
from src.gs_pino.data_freegs import FreeBndDataset

# 加载模型
checkpoint = torch.load("outputs/freegs_rhs_planet/best.pt", map_location="cuda")
model = PlaNetCore(
    n_measures=checkpoint["n_measures"],
    hidden_dim=checkpoint["args"]["hidden_dim"],
    nr=64,
    nz=64,
).to("cuda")
model.load_state_dict(checkpoint["model"])
model.eval()

# 准备输入
measures = torch.randn(1, 9).to("cuda")  # 9个输入参数
R = torch.randn(1, 64, 64).to("cuda")
Z = torch.randn(1, 64, 64).to("cuda")

# 预测
with torch.no_grad():
    pred = model((measures, R, Z))
```

### 可视化

```python
from src.gs_pino.visualize_freegs import plot_prediction

plot_prediction(
    pred=pred.cpu().numpy(),
    target=target.cpu().numpy(),
    R=R.cpu().numpy(),
    Z=Z.cpu().numpy(),
    output_path="prediction.png"
)
```

## 注意事项

1. **GPU强制**：训练脚本强制要求CUDA，确保使用GPU环境
2. **网格大小**：freegs生成时使用65×65，训练时自动插值到64×64
3. **课程学习**：前100个epoch仅MSE训练，之后才加入PDE约束
4. **数据归一化**：输入参数进行标准化处理（均值为0，标准差为1）
5. **早期停止**：验证损失50个epoch无改进时自动停止

## 参考文献

1. PlaNet-equil: Physics-informed neural operators for tokamak equilibrium reconstruction
2. Physics-Informed Neural Networks for the Grad-Shafranov Equation
3. freegs documentation: https://freegs.readthedocs.io/

## 许可证

MIT License
