# GS-PINO 算法技术详解：从 FNO 到 Transformer 的架构家族与两阶段物理约束训练

> 写给谁：想弄懂这套代码里"到底用了什么算法、为什么这么设计、各方法
> 什么关系"的初学者——不需要物理或深度学习的预先知识，但建议会一点
> Python 和基本微积分。
>
> 覆盖内容：Grad-Shafranov（GS）平衡求解问题；8 种网络架构（FNO、U-Net、
> UFNO、FNO-KAN、PI-DeepONet、POD-DeepONet、Transformer〔实验代号
> TKNO-lite〕、点式 KAN）各自的思想、结构、公式与实验结果；两阶段物理
> 约束（PINO）训练方法；以及整个实验系列（exp101–312）串起来的结论链。
>
> 对应代码：[src/gs_pino_fno_phys/models_alt.py](../src/gs_pino_fno_phys/models_alt.py)、
> [src/gs_pino/model 目录](../src/gs_pino_dn_fno_2608/model_dn_fno.py)、
> [train_pino.py](../src/gs_pino_fno_phys/train_pino.py)、
> [losses_pino.py](../src/gs_pino_fno_phys/losses_pino.py)。
> 完整实验档案见 [EXPERIMENTS.md](EXPERIMENTS.md)。

---

## 0. 一页速览

| 问题 | 给定 16 个标量（等离子体参数 + 线圈电流），预测整个二维磁通场 ψ(R,Z) |
|---|---|
| 数据 | data_v5/dn：MAST 双零位形，65×65 网格，2000/500/500 样本，训练用 N=500 |
| 训练 | 两阶段 PINO：先监督拟合（ψ + J），再物理微调（GS 残差 + Ip 约束） |
| 结果 | 最优 Transformer 算子 **0.635%** 相对 L2 误差；X 点定位 0.42 cm |

**一句话结论链**：纯局部卷积（U-Net）学不会磁面全局拓扑 → 谱域混合
（FNO）可行 → 多尺度谱（UFNO）更好 → 全局注意力（Transformer）最好。
决定精度的不是"用不用傅里叶"，而是**全局混合机制**本身。

---

## 1. 物理问题：Grad-Shafranov 方程

### 1.1 托卡马克与磁通

托卡马克（tokamak）是一种磁约束聚变装置：用强磁场把高温等离子体"装"在
真空室里。描述平衡态（等离子体静止时）磁场形状的核心方程是
**Grad-Shafranov（GS）方程**：

$$
\Delta^*\psi + \mu_0 R\, J_\phi(R,Z) = 0
$$

其中：

- $\psi(R,Z)$ 是**极向磁通**（poloidal flux，单位 Wb）——我们要预测的场。
  磁力线就是 ψ 的等值线，磁面（磁通面）的形状直接由 ψ 的等值线形状决定；
- $\Delta^*$ 是柱坐标下的"类拉普拉斯"算子
  $\Delta^* = \dfrac{\partial^2}{\partial R^2} - \dfrac{1}{R}\dfrac{\partial}{\partial R} + \dfrac{\partial^2}{\partial Z^2}$；
- $J_\phi$ 是环向电流密度（单位 A/m²）——与 ψ 通过等离子体剖面函数自洽
  （$J_\phi = R\,p'(\psi) + \dfrac{FF'(\psi)}{\mu_0 R}$）；
- $\mu_0 = 4\pi\times10^{-7}$ 是真空磁导率。

**free-boundary（自由边界）**：等离子体没有硬壁约束边界形状——等离子体
边界（separatrix，分离面）的位置和形状由**外部线圈电流**决定，而这些电流
正是我们网络的输入。这是本任务与"固定边界"求解的关键区别，也是它难的地方：
边界不是已知条件，是要从输入推断的输出。

> 出处：GS 方程见 Shafranov（1966）；入门教材：Freidberg, *Ideal
> Magnetohydrodynamics*（Plenum Press, 1987）。本仓库复现的 free-boundary
> 算子学习论文为 arXiv:2608.05555（其求解器与数据管线即 data/ 与
> gs_pino 包的来源）。

### 1.2 任务形式化

把 GS 求解变成**算子学习**（operator learning）问题：

$$
\Psi_\theta(\mathbf{c}) \approx \psi_\text{solution}(\mathbf{c})
$$

网络 $\Psi_\theta$ 输入 18 个通道（§2），输出 ψ 场（65×65 网格），一步到位，
不迭代求解（传统的 freegs/gspack 求解器要迭代几千步）。

**为什么值得做**：传统求解器单次约 1 秒（CPU），深度学习前向只要 1.6 ms
（GPU），快约 600 倍，可实时用于等离子体控制回路（10 kHz 级别）。

### 1.3 coil 分离（网络只学等离子体场）

总磁通 = 线圈产生的"真空场" + 等离子体电流产生的"等离子体场"：

$$
\psi_\text{total} = \psi_\text{plasma} + \sum_{k=1}^{11} I_k\, G_k(R,Z)
$$

其中 $G_k$ 是第 k 个线圈的 **Green 函数**（该线圈通单位电流时产生的场，
由解析公式给出，greens 恒等式验证误差 6e-8 Wb）。

**为什么网络只学 ψ_plasma**：线圈场占 |ψ_total| 幅值的 183%，且在线圈位置
有导体奇性——若让网络直接预测总场，物理残差 $\Delta^*\psi + \mu_0 R J$ 会被
线圈奇性主导，物理约束项完全失效。网络负责学"难的部分"（等离子体场），
线圈部分是解析可加的。评估时再通过上式把总场加回来（实现见
[evaluate_pino.py:144-146](../src/gs_pino_fno_phys/evaluate_pino.py#L144-L146)，
`einsum("kij,k->ij", greens, coil_currents)` 一步加回）。

---

## 2. 数据与输入表示

### 2.1 数据集

| 字段 | 值 |
|---|---|
| 数据 | `data_v5/dn/train.npz` 等（MAST 装置，双零位形 DN） |
| 网格 | 65×65（R、Z 各 64 个间隔） |
| 划分 | 2000 训练 / 500 验证 / 500 测试 |
| 本系列实验训练样本 | N=500（从 2000 池**嵌套子集**抽，perm seed 12345） |
| 参数范围 | 等离子体电流 Ip 0.3–0.8 MA、磁轴压强 paxis 1–5 kPa、fvac 0.3–0.8 等 |

**嵌套子集**（nested subset）：固定随机种子从全池抽 500 个，保证所有实验
（N=500 vs N=2000）训练样本是包含关系，方便公平对比与后续扩量。

**代码位置**：数据文件 `dn_fno_2608/data_v5/dn/{train,val,test}.npz`；
数据集与 z-score 实现见 [data_pino.py:42](../src/gs_pino_fno_phys/data_pino.py#L42)
`DNPinoDataset`（J 场重构 :67、扩展统计 :82、`__getitem__` :95）；
嵌套子集抽取 `nested_train_indices` 在
[data_dn_fno.py](../src/gs_pino_dn_fno_2608/data_dn_fno.py)。

### 2.2 18 个输入通道

| # | 通道 | 含义 | 类型 |
|---|---|---|---|
| 1 | R | 网格径向坐标（归一化到 [-1,1]） | 空间坐标场 |
| 2 | Z | 网格纵向坐标（[-1,1]） | 空间坐标场 |
| 3 | Ip | 等离子体电流 (A) | 标量（广播到全场） |
| 4 | paxis | 磁轴压强 (Pa) | 标量 |
| 5 | fvac | 真空通量函数 (Wb/m) | 标量 |
| 6 | alpha_m | 剖面形状指数 m | 标量 |
| 7 | alpha_n | 剖面形状指数 n | 标量 |
| 8–17 | I_P2U … I_P6L | 上下偏滤器 10 个线圈电流 (A) | 标量 |
| 18 | I_P1 | 中心螺线管电流 (A) | 标量 |

关键设计：**不给 X 点坐标、不给位形标签**——只给可测量量（电流 + 工程
参数）。位形识别（这个样本是 DN 还是 SN）是网络的隐含能力，由线圈电流
模式承载（exp010/011 已证实，config 通道冗余可删）。

### 2.3 归一化与目标

- 输入标量 z-score 归一化（统计量来自全训练池）；
- 目标 ψ_plasma 与 J 也 z-score 归一化（**z 域**训练）；
- 每个样本附带**核心 mask**（等离子体内部区域二值图）——J 在 mask 外恒为
  0，物理损失只在 mask 内计算（见 §5）。

### 2.4 评估指标

| 指标 | 含义 |
|---|---|
| rel L2 | $\|\hat\psi - \psi_\text{true}\|_2 / \|\psi_\text{true}\|_2$（z 域）——本文所有"0.635%" |
| GS 残差 | $\|\Delta^*\hat\psi + \mu_0 R \hat J\|$ 的 mask 内均值（物理自洽度，归一化 pde_scale） |
| Ip 相对误差 | $\|\sum(\hat J \cdot dA) - I_p\| / I_p$（电流积分约束） |
| X 点/O 点定位误差 | 分离面临界点位置误差（cm）——磁面拓扑的指纹 |

---

## 3. 前置知识速成（初学者必读）

以下每个概念只讲"解决什么问题 + 一个直觉"。用到的符号：$x$ 是输入场
(B, C, H, W)，$W$ 是权重。

### 3.1 卷积与感受野

卷积：滑窗加权求和。输出每个点只看输入的一个**小邻域**（如 3×3）。
**感受野**（receptive field） = 输出点能"看到"的输入区域大小。堆 N 层 3×3
卷积 + 池化（下采样），感受野按指数增长，但**远距离信息要经过很多层才
能传播**，且中间层会丢失细节（池化是丢信息的）。

> 直觉：卷积像"每个人只跟邻居商量"，全局共识要一级一级传。

### 3.2 傅里叶变换与谱卷积（FNO 的核心）

傅里叶变换把一张图拆成不同频率的正弦波叠加：$\mathcal{F}[x](k)$ 是频率 k
的分量（复数）。低频 ≈ 大尺度整体形状（磁面的大致轮廓），高频 ≈ 细节
（X 点附近的陡峭变化）。

**谱卷积**：在频率域把每个模态乘一个可学习系数 $R(k)$，再逆变换回空间：

$$
K x = \mathcal{F}^{-1}\!\big(R(k)\cdot \mathcal{F}[x]\big)
$$

- 傅里叶变换是 O(N log N)，而且**一次操作就让所有像素互相作用**（频域
  每个系数都来自全场的积分）——天然的"全局混合"；
- 只保留前 $m_1\times m_2$ 个低频模态（截断），既减少参数又是隐式的平滑
  先验（高频细节被砍掉）——FNO 的"模态数"就是这个截断数。

> 直觉：谱卷积像"开全员大会"——一次 FFT 就汇总了所有人的意见，按频率
> 分配权重（低频权重给大局观，高频给细节）。

### 3.3 MLP（多层感知机）

$h = W_2\,\sigma(W_1 x + b_1) + b_2$：逐点（per-pixel）的线性变换 + 非线性
激活。MLP **没有空间结构概念**——它对每个像素独立处理，除非把 (R,Z) 坐标
作为输入喂给它（DeepONet trunk 就是这么做的）。

### 3.4 Transformer 自注意力（Transformer 算子的核心）

自注意力让序列里每个 token 与**所有其他 token** 交互，权重由内容决定：

$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

- Q（查询）、K（键）、V（值）由输入线性投影而来；
- $QK^\top$ 是"每对 token 的相关性"矩阵（注意力权重）；softmax 归一化成
  概率；加权求和 V 得到新表示；
- 与谱卷积对比：谱卷积的混合权重 $R(k)$ 是**固定模态**的（所有样本共用），
  注意力权重是**内容自适应**的（每个样本、每对 token 单独算）——更灵活，
  但代价是二次复杂度 O(N²)（N 是 token 数），所以 65×65=4225 像素直接做
  注意力不可行，要先降采样成 32×32=1024 个 token（§4.7）。

### 3.5 SVD 与 POD（POD-DeepONet 的核心）

SVD 把矩阵 $X = U \Sigma V^\top$ 分解。对"样本 × 像素"的矩阵做 SVD，
右奇异向量 $V$ 的前 p 行就是数据的主导**模式**（主成分/POD 基）：
第一个模式抓住最大的变化方向，后面的逐次抓住剩余的最大方向。累计能量

$$
E_p = \frac{\sum_{k=1}^{p} \sigma_k^2}{\sum_{k} \sigma_k^2}
$$

告诉你有多少变化被前 p 个模式覆盖。POD-DeepONet 的整个想法：既然 ψ 解
空间可能低维，干脆用数据算出的前 p 个模式当"底座"，网络只学每个样本的
**系数**（§4.6）。

### 3.6 B-spline 基函数（KAN 的核心）

KAN（Kolmogorov-Arnold Networks）把神经元的"固定激活"换成**可学习的
样条函数**：每个连接一个一维函数

$$
\phi(x) = w_b\,\text{silu}(x) + w_s \sum_b c_b B_b(x)
$$

$B_b(x)$ 是 [−1,1] 上均匀结（knot）的分段多项式基（B-spline），系数 $c_b$
可学习——网络能学出任意的单变量函数形状（而 ReLU/GELU 是固定的）。
参数多（每个连接要学基系数），但理论上表达力强。

---

## 4. 架构家族

所有架构共享统一接口 `forward(x: (B,18,65,65)) -> (B,2,65,65)`（通道 0 =
ψ_plasma，通道 1 = J），通过 `MODEL_REGISTRY` 注册表接入训练管线，参数量
按"复数参数计 1 个"的约定统计。**两阶段训练方案对所有架构完全相同**，
所以架构对比是干净的（只变 `--model` 一个参数）。

### 4.1 FNO2d2608（基线，exp102）

```
x (18×65²)
  │ Conv2d 1×1 (lift, 18→64)
  ▼
[FNOBlock × 4]  ──每个块──►  SpectralConv2dR (谱卷积, 16×16 模态, 64→64)
  │                         + Conv2d 1×1 (逐点混合, 64→64)
  │                         → GELU
  ▼
Conv2d 1×1 (proj, 64→2)
y (2×65²)
```

**数学**：每个 FNOBlock 是

$$
h_{\ell+1} = \sigma\!\Big( \mathcal{F}^{-1}[ R_\ell(k)\, \mathcal{F}[h_\ell] ] + W_\ell h_\ell \Big)
$$

谱部分做全局混合（§3.2），1×1 卷积做逐点通道混合，GELU 是激活。

**关键代码**：[model_dn_fno.py](../src/gs_pino_dn_fno_2608/model_dn_fno.py) 的
`SpectralConv2dR`/`FNOBlock`。`SpectralConv2dR.forward` 里 `torch.fft.rfft2`
→ 只保留前 16×16 模态做复数乘法（`einsum "bixy,ioxy->boxy"`）→
`torch.fft.irfft2` 还原。

**参数**：约 4.21M（18 通道输入版，论文 9 通道口径 4,770,241）。

**结果**：exp102 两阶段 test rel L2 **0.800%**，Ip 误差 0.21%，mask 内 J
1.36%。单阶段做法（exp101，物理残差用数据 RHS）0.72%。

**为什么它有效**：椭圆方程（GS）的解具有"全局耦合"性质——任何一点的 ψ
由全场边界条件共同决定。FFT 一次操作实现全局混合，恰好匹配这个性质。

**位置与参考**

| 项 | 位置 |
|---|---|
| 源码 | [model_dn_fno.py:73](../src/gs_pino_dn_fno_2608/model_dn_fno.py#L73) `FNO2d2608`（谱核 `SpectralConv2dR` :27、`FNOBlock` :61、`build_model` :99） |
| 实验 | [exp102_pino_twostage_n500/](experiments/exp102_pino_twostage_n500/README.md)（两阶段基线 0.800%）；单阶段对照 [exp101_pino_rhs_n500/](experiments/exp101_pino_rhs_n500/README.md) |
| 脚本 | [run_exp101_102_pino.sh](scripts/run_exp101_102_pino.sh) |
| 参考 | Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021（arXiv:2010.08895） |

### 4.2 U-Net（纯卷积，exp301，失败）

```
        65² ────────────────────────► skip ───────────► 65² (DoubleConv + proj)
         │ ▲
   enc0 (24ch)                  dec0 (24ch)
   池化 2×                         上采样
         │ ▲
   enc1 (48ch) ┄┄ skip ┄┄► dec1 (48ch)
   池化 2×                        ...
         │ ▲
   enc2 (96ch) ┄┄ skip ┄┄► dec2 (96ch)
   池化 2×
         │ ▲
   enc3 (192ch) ─► bottleneck(192) ─► dec3 (192ch)
```

**思想**：编码器逐级下采样（信息压缩），解码器逐级上采样（细节恢复），
skip 连接把每级的空间细节直接送到对应解码级。CNN 的经典骨架。

**关键代码**：[models_alt.py](../src/gs_pino_fno_phys/models_alt.py) 的
`UNet2d2608`（2.76M 参数）。注意 `F.interpolate(size=...)` 用显式 size 而
非 scale_factor——65 不是 2 的幂，scale_factor 会取整错位。

**结果**：exp301 **2.202%**——FNO 的 2.75 倍。阶段1 卡在 3.017%（差 0.017
点达不到 3% 阈值，e300 兜底切换），X 点定位 1.83/2.32 cm（FNO 的 3-4 倍），
GS 残差 4.3 倍。所有"全局"指标同步退化。

**为什么失败（这是全系列最重要的对照）**：磁面拓扑（X 点位置、分离面
形状）由远处线圈电流 + 参数全局决定，3×3 卷积的感受野要堆十几层才够
远，而且池化把细节丢了。**局部感受野不足承载磁面全局耦合**。文献里
"CNN 在 GS 任务上最佳平衡"的结论在这里不迁移——因为该基准没有物理约束
项且是单阶段。

**位置与参考**

| 项 | 位置 |
|---|---|
| 源码 | [models_alt.py:140](../src/gs_pino_fno_phys/models_alt.py#L140) `UNet2d2608`（`_DoubleConv` :87、`_UpBlock` :103、构造器 `_build_unet2d2608` :181） |
| 实验 | [exp301_unet_pino_twostage_n500/](experiments/exp301_unet_pino_twostage_n500/README.md)（2.202% 失败归因） |
| 脚本 | [run_exp301_302_pino.sh](scripts/run_exp301_302_pino.sh) |
| 参考 | Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015（arXiv:1505.04597）；EXL-50U 五架构基准（arXiv:2608.23217，见 [ARCHS_SURVEY.md](ARCHS_SURVEY.md) §3） |

### 4.3 UFNO（多尺度谱 U-Net，exp302）

**思想**：U-Net 的编码器各层换成 FNOBlock，模态数随分辨率收缩——

```
65²: FNOBlock 16×16   ← 高分辨率谱核：保细节
32²: FNOBlock 16×16
16²: FNOBlock  8×8    ← 低分辨率谱核：承载全局
 8²: FNOBlock  4×4
```

解码器用普通卷积上采样 + skip。

**关键代码**：[models_alt.py](../src/gs_pino_fno_phys/models_alt.py) 的
`UFNO2d2608`（2.47M = FNO 的 0.59×）。谱核 `FNOBlock(w, m1, m2)` 的模态数
(m1,m2) 随层深度递减，与分辨率匹配。

**结果**：exp302 **0.729%**（FNO −9%），J mask 1.13%（−17%），X 点
0.52/0.63 cm。**全面小幅击败 FNO 且参数更少**。exp305 调优版（解码器也谱
化 + bottleneck ×2 + 模态 16→20）进一步到 0.7147%，主要收益是阶段1 收敛
翻倍（e70→e30）。

**为什么有效**：多尺度 = "高分辨率层保细节 + 低分辨率层承载全局"两个
能力解耦，比 FNO 单分辨率（16×16 模态一律作用于 65²）更细粒度。

**位置与参考**

| 项 | 位置 |
|---|---|
| 源码 | [models_alt.py:191](../src/gs_pino_fno_phys/models_alt.py#L191) `UFNO2d2608`（构造器 `_build_ufno2d2608` :246、调优版 `_build_ufno2d2608_tuned` :252）；仓库另有论文风格遗留实现 `UFNO2d`/`UFNO2d_v2`（[models.py:330-380](../src/gs_pino/models.py#L330-L380)，仅参考未用） |
| 实验 | exp302：[exp302_ufno_pino_twostage_n500/](experiments/exp302_ufno_pino_twostage_n500/README.md)；调优 exp305：[exp305_ufno_tuned_pino_twostage_n500/](experiments/exp305_ufno_tuned_pino_twostage_n500/README.md)；训练侧负收益 exp306：[exp306_ufno_weights_pino_twostage_n500/](experiments/exp306_ufno_weights_pino_twostage_n500/README.md) |
| 脚本 | [run_exp301_302_pino.sh](scripts/run_exp301_302_pino.sh)（exp302）；[run_exp305_306_pino.sh](scripts/run_exp305_306_pino.sh)（exp305/306） |
| 参考 | Wen et al., "U-FNO — An Enhanced Fourier Neural Operator-based Deep-learning Model for Multiphase Flow", Adv. Water Resour. 2022（arXiv:2109.03697） |

### 4.4 FNO-KAN（exp303，中性结果）

**思想**（来自 KANO 论文论点）：GS 是**变系数**方程（系数里含 R），固定
形状的激活可能表达力不足，逐点换成可学习激活（KAN）能补上。做法：FNO
骨架不变，把 FNOBlock 里的 1×1 卷积换成 **KANConv1x1**（逐像素 B-spline）：

$$
h_{\ell+1} = \sigma\Big( \mathcal{F}^{-1}[R_\ell(k)\mathcal{F}[h_\ell]] + \text{KAN}_{1\times1}(h_\ell) \Big)
$$

**关键代码**：[models_alt.py](../src/gs_pino_fno_phys/models_alt.py) 的
`KANConv1x1`（4.33M 参数，+3%）。三个工程细节：

1. **tanh 门控**：B-spline 基定义在 [−1,1]，而谱卷积输出无界——先
   $\tanh$ 压缩进样条域；
2. **融合矩阵乘法**：`h = silu(x)@Wb + B(x)@(Ws*C)` 一次矩阵乘法算完，
   绝不物化 (B, N, C_in, C_out) 的中间张量（会爆显存）；
3. **基函数在 no_grad 下计算**：B-spline 基只依赖输入 x 和固定网格，梯度
   只流向系数——省一半反向图，基的梯度路径本来就很小。

**结果**：exp303 **0.857%**——机制成立（收敛正常、无 NaN）但**增益为零**
（全部指标与 FNO 差 ≤0.1pp），代价是 4× 训练时长。KANO"变系数需可学习
激活"的论点在 65²×16×16 模态下未转化：R 因子的变化已经被谱核充分捕获。

> 与点式 KAN（§4.8，20.87% 失败）合起来是一个完整证据链：**KAN 可以
> 工作（插进 FNO 无损伤），但不更好**——瓶颈不在激活形状。

**位置与参考**

| 项 | 位置 |
|---|---|
| 源码 | [models_alt.py:270](../src/gs_pino_fno_phys/models_alt.py#L270) `KANConv1x1`（批量 B-spline 递归 `_bspline_bases_batched` :46）；`FNOKAN2d2608` :318、块 `_FNOKANBlock` :342、构造器 :355；B-spline 基函数从 [model_kan.py:38-114](../src/gs_pino_kan_2608/model_kan.py#L38-L114) 复用 |
| 实验 | [exp303_fnokan_pino_twostage_n500/](experiments/exp303_fnokan_pino_twostage_n500/README.md)（0.857% 中性结果） |
| 脚本 | [run_exp303_304_pino.sh](scripts/run_exp303_304_pino.sh) |
| 参考 | Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025（arXiv:2404.19756）；KANO 算子版（arXiv:2407.04192） |

### 4.5 PI-DeepONet（exp304，容量奇迹）

**思想**：DeepONet 把"参数 → 场"映射拆成两个 MLP 的内积：

$$
\psi(R,Z) = \sum_{k=1}^{p} b_k(\underbrace{I_p, p_\text{axis}, \ldots, I_{P1}}_{16\ \text{个标量}})\; \tau_k(R,Z)
$$

- **branch 网络**：16 个标量 → p=256 维系数向量；
- **trunk 网络**：(R,Z) 坐标 → 256 维基函数向量（学习的主干）；
- 输出 = 逐像素内积（einsum `"bp,bnp->bn"`）。

这就是"分支-主干分解"：标量输入→场输出问题的最直接结构（本问题与
DeepONet 的设定精确同构）。ψ 和 J 各一个 branch（J 是 ψ 的二阶导场，
单独建模更准）。

**关键代码**：[models_alt.py](../src/gs_pino_fno_phys/models_alt.py) 的
`PIDeepONet2d`。两个细节：trunk 末层权重 ×0.1（输出从小起步，避免初始
MSE 巨大）；branch 输入 = `x[:,2:].mean((2,3))`（标量通道是广播常数，
空间平均无损还原）。

**结果**：exp304 **0.802%**——0.6M 参数（FNO 的 1/7）达到与 FNO 统计上
不可区分的精度，训练最快（~20 min）。但 J mask 2.61%、Ip 0.384%（FNO
的约 1.8-2 倍）——**容量分配**：参数都被 ψ 主任务吃掉，J（二阶导场）
被饿着。exp307 把宽度 256→384（1.34M）验证了这个诊断：**0.7575%、Ip
0.210%（−45%，反超 UFNO）、J 2.13%**——加宽就修好了。

**位置与参考**

| 项 | 位置 |
|---|---|
| 源码 | [models_alt.py:381](../src/gs_pino_fno_phys/models_alt.py#L381) `PIDeepONet2d`（`_MLP` :366、构造器 :438、加宽版 `_build_pideeponet2d_wide` :444、傅里叶特征版 `_build_pideeponet2d_ff` :453）；仓库论文风格 PlaNet 系另有遗留（[models.py:199-275](../src/gs_pino/models.py#L199-L275)，仅参考未用） |
| 实验 | exp304：[exp304_pideeponet_pino_twostage_n500/](experiments/exp304_pideeponet_pino_twostage_n500/README.md)（容量奇迹）；加宽 exp307：[exp307_pideeponet_wide_pino_twostage_n500/](experiments/exp307_pideeponet_wide_pino_twostage_n500/README.md)；FF 失败 exp308：[exp308_pideeponet_ff_pino_twostage_n500/](experiments/exp308_pideeponet_ff_pino_twostage_n500/README.md) |
| 脚本 | [run_exp303_304_pino.sh](scripts/run_exp303_304_pino.sh)（exp304）；[run_exp307_308_pino.sh](scripts/run_exp307_308_pino.sh)（exp307/308） |
| 参考 | Lu et al., "Learning nonlinear operators via DeepONet…", Nature Machine Intelligence 3, 2021（arXiv:1910.03193）；SUNIST-2 物理约束 DeepONet 部署（调研见 [ARCHS_SURVEY.md](ARCHS_SURVEY.md) §3.3） |

### 4.6 POD-DeepONet（exp312，诚实负面）

**思想（与 DeepONet 只差一半）**：回顾 §4.5——DeepONet 的两半都是学出来
的：branch 学"这个样本的系数"，trunk 学"每个位置的基"。POD-DeepONet 的
关键发问：**trunk 的 256 个基函数，为什么不直接用统计方法从数据里算出
最优的那几个？** 于是把学习 trunk 换成**数据算出的固定 POD 基**：

$$
\psi_z(R,Z) = \psi_0(R,Z) + \sum_{k=1}^{p} b_k\, \phi_k(R,Z)
$$

- $\psi_0$：训练样本的平均场；$\{\phi_k\}$：样本快照矩阵的 SVD 右奇异向量
  （前 p 个，§3.5）；
- 基是固定 buffer（`register_buffer`，**无梯度**），阶段2 物理微调只能动
  系数 $b_k$——**低秩先验即实验假说**：若解的线性子空间足够，系数微调
  应达到学习 trunk 同档精度。

**SVD 怎么算的（具体数字）**：把 N=500 训练子集的 ψ 场排成
**500 × 4225** 矩阵（500 样本 × 65×65 像素），减去平均场后
`np.linalg.svd(X, full_matrices=False)` 得到 Vᵀ（4225×4225 的行是模式），
取前 p 行当基。注意**两个细节**（都踩过坑）：
① 基必须来自**嵌套训练子集**而不是全池（`idx = ds.indices`，[train_pino.py:55](../src/gs_pino_fno_phys/train_pino.py#L55)）——否则基和训练目标不同分布；
② 快照用 **z 域**（与网络训练目标一致，`pod_basis_from_dataset` 内部做了
z-score），不是物理域。

**p 怎么定：能量判据，不是调参**。累计能量

$$
E_p = \frac{\sum_{k=1}^{p}\sigma_k^2}{\sum_{k}\sigma_k^2},\qquad
p = \min\{p : E_p \ge 0.999\}
$$

按顺序累加奇异值平方占比，到 99.9% 就停（封顶 256 模态）。p 是
**数据结构的自动读数**，这正是本实验的巧妙处——它在回答"这个数据集
本质上需要多少个自由度"：

- $p_\psi = 3$（能量 99.924%）——ψ 解空间惊人地低秩；
- $p_j = 10$（能量 99.926%）——J 是 ψ 的二阶导场，低秩性弱约 3 倍。

**为什么会得到 p_psi=3（物理直觉）**：
1. **coil 分离**：线圈场被解析减掉，网络只学等离子体场——其形状自由度
   本来就少（真空场才是 11 维电流张出的）；
2. **椭圆方程解光滑**：GS 是椭圆型方程，解没有锐利结构，跨样本变化由
   几个大尺度方向主导（如整体幅值档位、上下不对称度、形状偏心率——SVD
   是纯统计分解，不保证每个模式有干净的名字，但量级就是"极少模式覆盖
   绝大多数变化"）；
3. **采样结构**：5 个标量参数 + 11 个电流在该解流形上张出的主导方向
   只有 3 个。

**网络只剩 branch**：16 标量 → 256×3 层 MLP → 13 个系数（ψ 的 3 + J 的
10），前向一步矩阵乘法 `psi = psi0 + branch(x) @ phi`，全部 **0.28M 参数**
（系列最小，exp307 的 0.21×）。训练 3.7 min（分支 13 系数，800 epoch 上限
远未用满）。

**结果**：exp312 **1.508%**（exp307 的 2 倍），X 点 0.85/0.85 cm（+0.26），
O 点 0.65 cm（+0.45），J 2.52%，Ip 0.253%（物理约束在低秩空间仍有效）。

**为什么失败：能量 ≠ 精度**（这个负面结果最有信息量）：
1. **SVD 按方差排序，不按物理重要性排序**。截断掉的 0.08% 能量恰是
   **X 点尺度的结构**——X 点附近陡峭变化占总方差极小（几个像素），但对
   磁面拓扑是决定性的。证据：误差集中在被截断的模态上（X 点 0.85 vs
   0.59 cm、O 点 0.65 vs 0.20 cm）。能量判据（0.999）与精度判据（0.7% 档）
   是两把不同的尺子；
2. **线性子空间的边界**：预测 ψ 被锁死在 3 个模式的线性组合里；而
   exp307 的学习 trunk 能让"基形状"随输入连续变化——本质上学习的是
   **非线性、自适应的基** ≈ "自适应非线性 POD"。它超出线性子空间的
   0.7pp 就是非线性表达的价值；
3. 结论：**"数据低秩 ⇒ 线性基够用"被否定**——低秩是数据的统计性质，
   不是解的完整描述。p_psi=3 本身是数据集结构诊断：v5/dn 的参数化采样
   在冗余维度上不会增加网络需学的模式（对未来数据增强/参数采样设计有
   直接意义）。

**什么时候 POD 反而有用（文献视角）**：POD-DeepONet（Lu et al. 2022）原
论文的卖点是**噪声鲁棒性**——固定基 = 强正则化，含噪数据下线性基天然
滤噪。本实验是干净数据 + 高精度要求，正则化变成瓶颈。未来若遇高噪声
数据（实验测量），POD 基可能反超。

**"POD 基 + 学习残差"双通道已在 exp313 落地**：本节的诊断（基管大体
形状、网络补 X 点细节）正是 exp313 的设计蓝本——低秩 POD 通道原样保留，
并联小 FNO 残差学 `y − ψ_pod`（分阶段冻结协议先 branch 后残差，消除通道
竞争）：**1.508% → 0.883%（−41%）**，X 点 0.85 → 0.54/0.56 cm、O 点
0.65 → 0.34 cm、GS 残差 −73%——被截断的 0.08% 能量确实是 X 点尺度结构，
且谱卷积残差能把它学回来。870K 参数（FNO 的 0.21×）达到 FNO 的 ~90%
精度（median 持平 0.645 vs 0.642，X 点反超）；代价是 Ip 略退化
（0.253% → 0.338%：残差放大 J 自由度，分布更准但积分匹配略松）。详见
[exp313 README](experiments/exp313_pod_residual_pino_twostage_n500/README.md)。

**残差通道需要多少全局性？**（exp314 消融，残差换成纯卷积 UNet）——
POD 基承担全局形状后，基外细节**部分**需要全局混合：纯卷积残差把 POD
从 1.508% 拉到 1.099%（−27%：X 点 −25%、O 点 −44%、GS 残差 −62%），但
明显弱于 FNO 残差（+0.22pp）——完整任务下纯卷积 = FNO 的 2.75× 差距，
残差任务下缩小到 ~1.25×，谱混合仍提供 X 点 −0.1 cm 级别的边际增益。
详见 [exp314 README](experiments/exp314_pod_residual_unet_pino_twostage_n500/README.md)。

**位置与参考**

| 项 | 位置 |
|---|---|
| 源码 | [models_alt.py:550](../src/gs_pino_fno_phys/models_alt.py#L550) `pod_basis_from_snapshots`（SVD + 能量判据）；`PODDeepONet2d` :581、构造器 :629；exp313 `PODResidual2d2608` :641（POD 通道 + FNO 残差 + 冻结接口 `set_branch_frozen`/`set_residual_frozen`）；训练侧基计算 `pod_basis_from_dataset`（[train_pino.py:55](../src/gs_pino_fno_phys/train_pino.py#L55)，注意基必须来自嵌套子集 `ds.indices`）；冻结协议 `--pod-pretrain-epochs`（[train_pino.py:151](../src/gs_pino_fno_phys/train_pino.py#L151)，交接 :290、阶段1 挂起 :299） |
| 实验 | [exp312_pod_deeponet_pino_twostage_n500/](experiments/exp312_pod_deeponet_pino_twostage_n500/README.md)（1.508% 诚实负面 + 能量诊断）；[exp313_pod_residual_pino_twostage_n500/](experiments/exp313_pod_residual_pino_twostage_n500/README.md)（0.883%，双通道分工 + 冻结协议）；[exp314_pod_residual_unet_pino_twostage_n500/](experiments/exp314_pod_residual_unet_pino_twostage_n500/README.md)（1.099%，UNet 残差消融——基外细节部分需要全局混合） |
| 脚本 | [run_exp311_312_pino.sh](scripts/run_exp311_312_pino.sh)（exp312）；[run_exp313_pino.sh](scripts/run_exp313_pino.sh)（exp313）；[run_exp314_pino.sh](scripts/run_exp314_pino.sh)（exp314，均 `--pod-pretrain-epochs 60`） |
| 参考 | Lu et al., "A comprehensive and fair comparison of two neural operators…", 2022（arXiv:2304.00643，POD-DeepONet） |

### 4.7 Transformer 算子（TKNO-lite，exp311，系列最优）

**思想**：这个架构的实质是一个 **conv-stem Transformer**——卷积主干保
全分辨率细节 + 全局自注意力做混合。它来自 TKNO（Transformer-KAN Neural
Operator，[arXiv:2511.19114](https://arxiv.org/abs/2511.19114)——在**同
一个 GS 任务**（EXL-50U，五架构基准）上胜出，监督 0.25%），但按 exp303
证据**剥离了 KAN**（逐点 KAN 增益 ≈ 0）。所以实现里**没有 KAN**，本质
就是纯 Transformer；"TKNO-lite"只是实验代号（lites = 去掉 KAN 的 TKNO）。
结构：

```
x (18×65²)
  │ Conv3×3 (stem, 18→64) + GELU      ← 65² 细节路径
  ▼
stem (64×65²)
  │ Conv4×4/s2 (patch embed, 64→128) + 位置编码
  ▼
tokens (128×32² = 1024 token)
  │ [TransformerBlock × 4]  ──每个块──►  LayerNorm → 8头自注意力
  │                                      LayerNorm → MLP(GELU, 4×宽)
  ▼
bilinear 上采样 (size=65², 显式)
  │ 与 stem 拼接 (128+64 → 64, DoubleConv)
  ▼
Conv1×1 → y (2×65²)
```

几个设计决策（都有明确理由）：

- **为什么 32²=1024 token**：65²=4225 直接自注意力是 O(N²)，batch 16 下
  不可行；1024 是 8 的倍数，能走 fused SDPA 快速路径（33²=1089 会退化）；
- **为什么保留 conv stem 的 65² skip**：patch 降采样会丢 X 点尺度的细节，
  用 stem 的原始分辨率做 skip 拼接补回（跟 U-Net 一个道理）；
- **位置编码**：注意力本身对 token 顺序不敏感，加可学习的 2D 位置编码
  告诉它"这是哪个网格位置"；
- **`need_weights=False`**：默认会物化 (B, heads, N, N) 注意力矩阵（4.6
  GiB），关闭后走 fused SDPA（0.92 GiB）。

**结果**：exp311 **0.635%**——系列新纪录（比上一位 exp305 低 11%），且
只有 1.21M 参数（FNO 的 0.29×）。全指标最佳：J mask 0.977%（UFNO −14%）、
Ip 0.189%、X 点 0.42/0.43 cm、O 点 0.18 cm。代价：训练 80 min（本 torch
构建无 flash 注意力，走 mem-efficient 后端，是 FNO 的 5.2×）——纯时间
代价，不改变结论。

**为什么它最强**：与 exp301（UNet 失败）/exp302（谱系成功）对照——
**磁面全局耦合的决定变量是混合的全局性，不是具体实现**。注意力的
内容自适应混合比谱卷积的固定模态混合更贴合局部结构（J 的 −14% 就是
证据：J 是二阶导场，对细节敏感）。

**位置与参考**

| 项 | 位置 |
|---|---|
| 源码 | [models_alt.py:491](../src/gs_pino_fno_phys/models_alt.py#L491) `TKNOlite2d2608`（块 `_TransformerBlock` :467、构造器 `_build_tkno_lite` :539）；代码注册表键名 `tkno_lite`（[models_alt.py:654](../src/gs_pino_fno_phys/models_alt.py#L654)） |
| 实验 | [exp311_tkno_lite_pino_twostage_n500/](experiments/exp311_tkno_lite_pino_twostage_n500/README.md)（0.635% 系列最优） |
| 脚本 | [run_exp311_312_pino.sh](scripts/run_exp311_312_pino.sh) |
| 参考 | TKNO（Transformer-KAN Neural Operator），ICML 2026（arXiv:2511.19114，同一 GS 任务五架构基准胜出）；Vaswani et al., "Attention Is All You Need", NeurIPS 2017（arXiv:1706.03762） |

### 4.8 点式 KAN（kan 轨，20.87% 失败，对照）

独立实验轨（`gs_pino_kan_2608`）：论文口径的纯逐点 KAN
（19 通道 → 10 → 2，每连接一个 B-spline 函数）。

**结果**：20.87%——两个原因叠加：① 只有 1,260 参数（容量不足）；
② **逐点结构没有空间混合**——每个 (R,Z) 像素只看到自己的输入，磁面
全局耦合完全无从谈起。这是"架构对比"系列没做之前就得到的教训，也是
FNO-KAN（§4.4）要保留谱卷积的原因。

**位置与参考**

| 项 | 位置 |
|---|---|
| 源码 | [model_kan.py:197](../src/gs_pino_kan_2608/model_kan.py#L197) `BSplineKAN`（`KANLayer` :135、`_knots` :38、`_bspline_bases` :47、解析二阶导 `forward_with_derivs` :223、剪枝 :242）；训练脚本 [train_kan.py](../src/gs_pino_kan_2608/train_kan.py)、评估 [evaluate_kan.py](../src/gs_pino_kan_2608/evaluate_kan.py) |
| 实验 | [kan/experiments/exp001_kan_v5/model_b19ch_kan_mix](../dn_fno_2608/kan/experiments/exp001_kan_v5/model_b19ch_kan_mix/)（20.87% 失败） |
| 脚本 | 见 [train_kan.py:38](../src/gs_pino_kan_2608/train_kan.py#L38) 用法示例 |
| 参考 | "基于 Kolmogorov-Arnold Networks 的 Grad-Shafranov 方程自由边界问题求解方法", Acta Phys. Sin. 75(2026)150504, DOI 10.7498/aps.75.20260331（逐点 KAN 原文，本轨复现） |

### 4.9 参数-精度全景

| 架构 | 参数 | test rel L2 | 一句话 |
|---|---|---|---|
| 点式 KAN | 1.3k | 20.87% | 容量 + 无全局混合，双败 |
| U-Net | 2.76M | 2.202% | 纯局部卷积，全局耦合学不会 |
| POD-DeepONet | 0.28M | 1.508% | 低秩先验过度激进 |
| FNO | 4.21M | 0.800% | 基线：谱卷积全局混合 |
| PI-DeepONet | 0.60M | 0.802% | 容量奇迹，J/Ip 通道饿着 |
| FNO-KAN | 4.33M | 0.857% | KAN 可工作但不更好 |
| DeepONet wide | 1.34M | 0.7575% | 加宽修复 J/Ip 短板 |
| UFNO | 2.47M | 0.729% | 多尺度谱，参数只有 FNO 0.59× |
| UFNO tuned | 3.60M | 0.7147% | 解码器谱化，边际小胜 |
| **Transformer**（TKNO-lite） | **1.21M** | **0.635%** | **全局注意力，系列最优** |

---

## 5. 两阶段 PINO 训练方法

**训练方法代码地图**（§5 全程对应）：

| 内容 | 位置 |
|---|---|
| 两阶段主循环（stage/ramp/损失组合） | [train_pino.py:218-315](../src/gs_pino_fno_phys/train_pino.py#L218-L315) |
| 阶段切换判据 | [train_pino.py:277-282](../src/gs_pino_fno_phys/train_pino.py#L277-L282) |
| 物理损失实现（Δ\*、残差、Ip） | [losses_pino.py:37-78](../src/gs_pino_fno_phys/losses_pino.py#L37-L78)（`lap_star_torch` :37、`pde_residual_rhs` :48、`pde_residual_self` :60、`ip_constraint` :71） |
| 模型注册表/统一接口 | [models_alt.py:640-661](../src/gs_pino_fno_phys/models_alt.py#L640-L661)（`MODEL_REGISTRY` + `build_model`） |
| 训练入口/超参 | [train_pino.py:111-151](../src/gs_pino_fno_phys/train_pino.py#L111-L151) |
| 评估与 coil 分离加回 | [evaluate_pino.py:57-201](../src/gs_pino_fno_phys/evaluate_pino.py#L57-L201)（`load_checkpoint` :57、psi_total 重建 :144） |

### 5.1 为什么纯监督不够？

纯 MSE 训练（exp011）只能学到"数据里有的"：
- 它学不到 **psi↔J 自洽**：数据里的 J 是求解器算出来冻着的，ψ 和 J 的
  关系不会被监督强化，预测对组合可能物理不自洽；
- 它学不到 **Ip 约束**：J 场积分等于等离子体电流——数据里有这个信息，
  但网络不会自动尊重积分守恒。

物理损失（PINO，Physics-Informed Neural Operator）把 GS 方程本身写进
损失函数，让网络学会"满足方程"——**纯数据管线拿不到的自洽性与积分
约束，物理损失提供**（exp101/102 的直接证据：Ip 误差 0.21%，mask 内
J 1.36%）。

### 5.2 两阶段损失（做法2，exp102 起）

**阶段 1（监督拟合）**——先把场拟合好：

$$
L_1 = \underbrace{\text{MSE}(\hat\psi_z,\ \psi_z)}_{\text{ψ 主任务}}
      + w_j \cdot \underbrace{\text{masked MSE}(\hat J_z,\ J_z)}_{\text{core mask 内}}
$$

**阶段 2（物理微调）**——在 L1 上追加自洽残差与 Ip 约束：

$$
L_2 = L_1 + \lambda(t)\Big[
      \underbrace{w_\text{pde}\, \big\|\tfrac{\Delta^*\hat\psi + \mu_0 R \hat J}{\text{pde\_scale}}\big\|^2_\text{mask内}}_{\text{GS 自洽残差（用网络自己的 J）}}
      + \underbrace{w_\text{ip}\, \big\|\tfrac{\sum \hat J\, dA - I_p}{\text{ip\_scale}}\big\|^2}_{\text{等离子体电流积分约束}}
      \Big]
$$

要点：

1. **自洽残差用的是网络自己的 J**（`pde_residual_self`），不是数据 RHS——
   这迫使 ψ 与 J 互相一致（做法1/exp101 用数据 J 当 RHS，只能做输出侧
   平滑，给不了自洽）；
2. **mask 内**：残差只在等离子体核心区内算（mask 外 J 恒为 0，二阶差分
   会有谱振铃假象）；
3. **pde_scale / ip_scale 归一化**：把不同物理量纲的项拉到 O(1)（0.362
   Wb/m²、5.51e5 A），否则数值上没法平衡权重；
4. **权重**：$w_\text{pde}=0.1,\ w_\text{ip}=1.0,\ w_j=1.0$（exp306 实验
   证明这组已经吃满杠杆，加大反而挤压 ψ 主任务）；
5. **λ(t) 是 warm-up ramp**：见下。

### 5.3 为什么必须有 warm-up ramp？（踩过的坑）

阶段切换的瞬间，psi↔J 自洽**尚未建立**：J 只是监督拟合的，Δ*ψ + μ0RJ
残差很大。若物理权重直接全量上（λ=1），巨大的物理梯度会把阶段1 辛苦
拟合的 ψ 场摧毁——第一版实现（v6）阶段2 首 epoch 直接爆炸。

修复：物理权重从 0 **线性预热** 30 个 epoch（`ramp = min(1, (epoch −
switch)/30)`）——给网络一个"过渡期"，在残差还很大的时候物理项只轻轻
推，等 ψ↔J 开始自洽了再全力拉。

### 5.4 阶段切换判据

$$
\text{stage 1} \to \text{stage 2}:\quad \text{val rel L2} < 3\% \quad\text{或}\quad \text{epoch} \ge 300
$$

先达标者触发（300 是兜底）。各架构切换时机：FNO e29、DeepONet e42、
UFNO e70（调优版 e30）、Transformer e66、POD e16（线性基起步最快）。

### 5.5 物理残差的计算细节

`lap_star_torch` 用**二阶中心差分**在 interior (63×63) 网格上离散 Δ*：

$$
\frac{\partial^2 \psi}{\partial R^2} \approx \frac{\psi_{i+1,j} - 2\psi_{i,j} + \psi_{i-1,j}}{dR^2},
\qquad
\frac{\partial \psi}{\partial R} \approx \frac{\psi_{i+1,j} - \psi_{i-1,j}}{2\,dR}
$$

$R_c$ 取 interior 网格的物理 R 值（防除零加 1e-8）。J 场的预测在**物理
域**（反 z-score）参与残差——物理损失永远作用在物理量上（Wb、A/m²），
监督项留在 z 域。

### 5.6 完整训练超参

| 超参 | 值 | 说明 |
|---|---|---|
| 优化器 | AdamW lr 1e-3, wd 1e-4 | |
| 调度器 | ReduceLROnPlateau（patience 20, factor 0.5, min_lr 1e-5） | 按 val rel L2 |
| batch / epochs | 16 / 800 | |
| 早停 | patience 75，**仅阶段2 起计** | 阶段1 不早停 |
| 权重 | w_pde=0.1, w_ip=1.0, w_j=1.0, ramp 30 epochs | §5.2 |
| 阶段1 | 阈值 3% / 兜底 300 epochs | §5.4 |
| 随机性 | seed 1（训练）、perm_seed 12345（子集抽取） | 全系列固定，保证可比 |

**phase-aware best**：阶段1 与阶段2 各记一个最佳 val 权重存档，artifact
优先取阶段2 best（阶段2 一旦开始，阶段1 的模型不算数）。

### 5.7 训练循环伪代码

```
for epoch in 1..800:
    stage = 1 if switch_epoch is None else 2
    ramp  = min(1, (epoch - switch_epoch) / 30) if stage == 2 else 0
    for batch:
        pred = model(x)
        loss = MSE(psi_z) + w_j * maskedMSE(J_z)        # 阶段1 一直有
        if stage == 2:
            loss += ramp * (w_pde * pde_residual_self(ψ, J, mask)
                          + w_ip * ip_constraint(J, mask, Ip))
        loss.backward(); optimizer.step()
    if stage==1 and (val_rel_l2 < 3% or epoch == 300):
        switch_epoch = epoch                            # 进入阶段2
    if stage==2 and no improvement for 75 epochs:
        early stop
```

---

## 6. 结果全景与结论链

### 6.1 总结果表（test n=500，data_v5/dn）

| 实验 | 架构 | rel L2 total | J mask | Ip | X 点 (cm) |
|---|---|---|---|---|---|
| exp301 | U-Net | 2.202% | 3.86% | 0.587% | 1.83/2.32 |
| exp312 | POD-DeepONet | 1.508% | 2.52% | 0.253% | 0.85/0.85 |
| exp314 | POD 基 + UNet 残差 | 1.099% | 2.33% | 0.403% | 0.64/0.64 |
| exp313 | POD 基 + 小 FNO 残差 | 0.883% | 1.76% | 0.338% | 0.54/0.56 |
| exp102 | FNO | 0.800% | 1.36% | 0.21% | — |
| exp304 | PI-DeepONet | 0.802% | 2.61% | 0.384% | 0.73/0.70 |
| exp303 | FNO-KAN | 0.857% | 1.45% | 0.238% | 0.67/0.71 |
| exp306 | UFNO 训练调优 | 0.7794% | 1.14% | 0.240% | — |
| exp307 | DeepONet wide | 0.7575% | 2.13% | 0.210% | 0.59/0.51 |
| exp302 | UFNO | 0.729% | 1.13% | 0.213% | 0.52/0.63 |
| exp305 | UFNO tuned | 0.7147% | 1.20% | 0.182% | 0.63/0.61 |
| **exp311** | **Transformer**（TKNO-lite） | **0.635%** | **0.977%** | **0.189%** | **0.42/0.43** |

（X 点列缺失 = 该实验未报；完整每桶数据见各实验 README 与
[EXPERIMENTS.md](EXPERIMENTS.md)。）

### 6.2 结论链（按证据顺序）

1. **纯局部卷积失败**（exp301 vs exp102）：U-Net 2.202% = FNO 的 2.75×，
   全局指标（X 点、GS 残差、J）同步退化 → 局部感受野不足承载磁面全局
   拓扑。文献"CNN 最佳"结论不迁移（该基准无物理约束/单阶段）。
2. **全局混合是决定变量，不是傅里叶**（exp301/302/311 三件套）：谱卷积
   （FNO/UFNO）与全局注意力（Transformer）都成功、纯局部都失败 →
   "混合的全局性"才是决定因素；注意力甚至优于谱（内容自适应 vs 固定
   模态）。**Transformer 算子 0.635%（exp311）是当前系列最优**。
3. **多尺度谱 > 单尺度谱**（exp302 vs exp102）：高分辨率保细节 + 低分辨
   率承载全局的解耦，参数还少 41%。
4. **KAN 可工作但不更好**（exp303 + 点式 KAN 轨）：插进 FNO 增益 ≈ 0
   （变系数论点被谱核吸收），逐点 KAN 双败（容量 + 无全局）。结论：
   别为 KAN 付 4× 训练时间。
5. **分支-主干分解是最经济的局部解**（exp304/307）：0.6M 参数追平 4.2M
   FNO，加宽到 1.34M 反超（0.7575%）；J/Ip 短板是容量分配问题，加宽即愈。
6. **低秩先验单独封顶，但可修复**（exp312 → exp313/314）：p_psi=3（99.92%
   能量）封顶 1.5%；能量判据 ≠ 精度判据；p_psi=3 是数据集结构诊断（v5/dn
   解空间极低秩）。**并联残差通道直接修复**：小 FNO 残差（exp313）1.508%
   → 0.883%（−41%）、X 点 0.85 → 0.54/0.56 cm、O 点 0.65 → 0.34 cm——被
   截断的 0.08% 能量确实是 X 点尺度结构；870K 参数（FNO 的 0.21×）达 FNO
   ~90% 精度。**残差架构消融（exp314）**：纯卷积 UNet 残差 1.099%（−27%，
   弱 0.22pp）——基外细节"部分需要"全局混合（完整任务 2.75× → 残差任务
   1.25× 差距），谱混合仍是最优残差。分工靠分阶段冻结协议（先 branch 60
   epochs 后残差），消除双通道竞争。
7. **物理损失的价值**（exp101/102 系列）：自洽 + Ip 约束是纯监督给不了
   的；物理权重必须 warm-up（λ 从 0 线性爬升 30 epochs），否则阶段2 首
   epoch 爆炸。

### 6.3 已知失败方向（省钱清单）

| 方向 | 实验 | 结论 |
|---|---|---|
| 损失权重加大 | exp306 | 杠杆已吃满，加大负收益 |
| trunk 傅里叶特征 | exp308 | 高频注入被二阶差分放大，梯度风暴（机制性不兼容） |
| POD 低秩基（独立） | exp312 | 3 模态封顶 1.5%，线性子空间装不下 X 点细节；**混合路线可行**（exp313：并联 FNO 残差 → 0.883%） |
| 点式 KAN | kan 轨 | 1.3k 参数 + 无全局混合，20.87% |
| FNO-KAN | exp303 | 机制成立、增益零、4× 时间 |

---

## 7. 复现指南

```bash
# 两阶段训练 + 评估（exp311 Transformer / exp312 POD-DeepONet）
bash dn_fno_2608/scripts/run_exp311_312_pino.sh all

# 任意架构（注册表内：fno2d2608 / unet / ufno / ufno_tuned / fnokan /
#             deeponet / deeponet_wide / deeponet_ff / tkno_lite / pod_deeponet）
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model tkno_lite \
  --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz \
  --n-train 500 --seed 1 --epochs 800 \
  --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp311_tkno_lite_pino_twostage_n500
```

产物约定（每实验目录）：`best.pt`（含 `model` 键与 `pod_basis`）、
`history.json`（逐 epoch 损失/val）、`args.json`、`metrics.json`、
`figures/`（fig1 best/worst 场对比、fig2 指标分布、fig3 几何定位 +
`stats_per_sample.json`）、`train.log`/`eval.log`。

新架构接入：在 `models_alt.py` 实现 `forward(x) -> (B,C,65,65)` + `count_params()`，
注册进 `MODEL_REGISTRY`，训练管线零改动（两阶段逻辑与骨架完全解耦）。

---

## 8. 术语表

| 术语 | 含义 |
|---|---|
| GS 方程 | Grad-Shafranov：托卡马克平衡的核心偏微分方程 |
| free-boundary | 自由边界：边界形状由线圈电流决定而非给定 |
| 磁通 ψ / 磁面 | 极向磁通场；等值线 = 磁力线 = 磁面 |
| X 点 / O 点 | 分离面的临界点：X 形分岔点 / 磁轴中心点 |
| coil 分离 | 总场 = 等离子体场 + Σ 线圈电流×Green 函数，网络只学前者 |
| 模态（modes） | 谱卷积保留的低频傅里叶分量数（16×16 等） |
| 感受野 | 输出像素能看到的输入区域 |
| 谱卷积 | 频域乘可学习系数再逆变换 = 全局混合核 |
| 自注意力 | token 间内容自适应的全局加权求和 |
| z-score | 减均值除标准差，把量纲不同的量拉平 |
| 嵌套子集 | 固定种子抽取的子集，跨 N 实验可比较 |
| mask | 等离子体核心区二值图；物理损失只在 mask 内 |
| POD/SVD | 数据主导模式分解；低秩基的来源 |
| B-spline | 分段多项式基，KAN 的可学习激活 |
| 两阶段 PINO | 阶段1 监督拟合 → 阶段2 物理微调 |
| warm-up ramp | 物理权重从 0 线性爬升，防止阶段切换爆炸 |
| pde_scale / ip_scale | 物理量归一化尺度（0.362 Wb/m²、5.51e5 A） |
| 早停 patience | 验证指标连续 N 个 epoch 无改进就停（75） |
| phase-aware best | 阶段1/2 各存最佳权重，artifact 优先阶段2 |

---

## 9. 延伸阅读

- 系列总览：[EXPERIMENTS.md](EXPERIMENTS.md)（全部实验台账与结论链）
- 架构选型调研：[ARCHS_SURVEY.md](ARCHS_SURVEY.md)（文献证据 + 候选矩阵）
- 数据说明：[data_v5/README.md](data_v5/README.md)
- 每实验细节：`experiments/expXXX_*/README.md`（结果表 + 归因分析）
- 文献锚点：FNO（Li et al. 2021）、DeepONet（Lu et al. 2021）、POD-DeepONet
  （Lu et al. 2022）、TKNO（arXiv:2511.19114，同一 GS 任务）、TCV PINO
  部署（arXiv:2606.09487）、FNO 自由边界 DN（arXiv:2608.05555，本仓库
  复现的论文）
