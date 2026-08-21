---
puppeteer:
  format: A4
  scale: 0.85
---

<style>
.markdown-preview-view .markdown-preview-section {
  width: 860px !important;
  margin: 0 auto !important;
  padding: 0 2rem !important;
  min-width: unset !important;
  max-width: unset !important;
}
pre,table {overflow-x:auto;}
</style>

<div align="center">

# 基于物理信息神经算子的 Grad-Shafranov 方程快速求解方法研究

陈佳林¹

<p>1. 中国聚变能源有限公司，上海，201102</p>

</div>
 

&emsp;&emsp;**摘要**：托卡马克等离子体平衡的实时重建对等离子体控制与位形优化具有重要意义，其核心是求解自由边界 Grad-Shafranov（GS）方程。传统数值求解器基于 Picard 迭代与 Green 函数方法，单次求解需数秒，难以满足实时需求。本文提出一种基于物理信息神经算子（Physics-Informed Neural Operator, PINO）的 GS 方程快速求解方法，采用傅里叶神经算子架构，以网格坐标、5 个等离子体参数和 11 个线圈电流共 18 个输入通道为输入，端到端输出等离子体极向磁通分布，并通过 Green 函数解析叠加线圈场获得全域通量。针对物理信息训练设计了两种方案：做法 1 为单阶段残差约束，将预测场的修正拉普拉斯算子拉向数据源项；做法 2 为两阶段自洽训练，先联合监督通量与环向电流密度，再引入通量-电流自洽残差和等离子体电流积分约束，并采用 30 轮线性预热权重避免阶段切换时发散。在 MAST 装置双零位形数据集上的实验表明：加入物理残差后，相对 L2 误差由纯数据驱动的 0.84% 降至单阶段 0.72% 与两阶段 0.80%；两阶段方法同时实现通量-电流自洽，等离子体内电流密度误差 1.36%，等离子体电流约束相对误差 0.21%，这是纯数据管线无法获得的能力。在双零与单零位形混合数据上，同一模型无需位形标签即可同时求解两种位形，整体相对 L2 误差 0.70%，与单一位形训练相当。训练后的模型单次推理约 8.6 毫秒，较传统求解器加速两个数量级以上。结果表明，物理信息神经算子方法可在保证精度的前提下实现自由边界 GS 方程的毫秒级求解，为托卡马克平衡实时重建提供了可行的技术路径。

&emsp;&emsp;**关键词**: Grad-Shafranov 方程；物理信息神经算子；傅里叶神经算子；两阶段训练；托卡马克平衡

&emsp;&emsp;**中图分类号**: TL612

&emsp;&emsp;**文献标志码**: A

 

---

 

<div align="center">

## A Physics-Informed Neural Operator Approach for Fast Solution of the Grad-Shafranov Equation

Jialin Chen¹

<p>1. China Fusion Energy Co., Ltd., Shanghai, 201102</p>

</div>
 

 

&emsp;&emsp;**Abstract**: Real-time reconstruction of tokamak plasma equilibrium is essential for plasma control and configuration optimization, and requires fast solution of the free-boundary Grad-Shafranov (GS) equation. Classical numerical solvers based on Picard iteration and Green's function methods require seconds per solve, which limits real-time applicability. This paper presents a physics-informed neural operator (PINO) approach for fast GS equilibrium computation. The model employs a Fourier neural operator architecture that maps 18 measurable quantities—grid coordinates, five plasma parameters, and eleven coil currents—directly to the plasma poloidal flux distribution, with the coil field added back analytically via Green's functions. Two physics-informed training schemes are developed: scheme 1 applies a single-stage residual constraint that pulls the modified Laplacian of the predicted flux toward the data source term; scheme 2 employs two-stage self-consistent training that first jointly supervises flux and toroidal current density, then introduces a flux–current self-consistency residual and a plasma current integral constraint, with a 30-epoch linear weight warm-up to prevent divergence at stage transition. Experiments on 500 double-null equilibria of the MAST device show that incorporating the physics residual reduces the relative L2 error from 0.84% for the data-only baseline to 0.72% with scheme 1 and 0.80% with scheme 2. Scheme 2 additionally achieves flux–current self-consistency with 1.36% current density error within the plasma and plasma current conservation with 0.21% relative error, capabilities unattainable by pure data-driven pipelines. On mixed double-null and single-null data, the same model solves both configurations without configuration labels, reaching an overall relative L2 error of 0.70% comparable to single-configuration training. The trained model performs inference in approximately 2 milliseconds, more than two orders of magnitude faster than classical solvers. These results demonstrate that physics-informed neural operators offer a viable pathway toward millisecond-scale free-boundary GS equilibrium computation for real-time tokamak applications.

&emsp;&emsp;**Key words**: Grad-Shafranov equation; physics-informed neural operator; Fourier neural operator; two-stage training; tokamak equilibrium

&emsp;&emsp;**CLC number**: TL612

 