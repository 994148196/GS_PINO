#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 PPT 生成算法架构示意图（9 张，200 dpi PNG，适合 16:9 页面）。

用法:
    C:/Users/HP/.conda/envs/torch5060/python.exe paper/make_ppt_arch_diagrams.py

输出: paper/ppt_assets/*.png

配色约定（全系列统一）:
  灰=输入/输出  蓝=卷积/编码器  橙=谱卷积(FFT)  紫=注意力  绿=MLP/分支
  红(斜纹)=冻结/固定(POD 基)  黄=重采样/嵌入/训练技巧  青=KAN  紫罗兰=残差通道
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "ppt_assets"
OUT.mkdir(exist_ok=True)

GRAY = ("#ECEFF1", "#546E7A")     # 输入/输出
BLUE = ("#BBDEFB", "#1565C0")     # 卷积
BLUEL = ("#E3F2FD", "#1565C0")    # 容器
ORANGE = ("#FFE0B2", "#E65100")   # 谱卷积
PURPLE = ("#E1BEE7", "#6A1B9A")   # 注意力
PURPLEL = ("#F3E5F5", "#6A1B9A")
GREEN = ("#C8E6C9", "#2E7D32")    # MLP
RED = ("#FFCDD2", "#B71C1C")      # 冻结/固定
YELLOW = ("#FFF9C4", "#F9A825")   # 重采样/嵌入
TEAL = ("#B2DFDB", "#00695C")     # KAN
VIOLET = ("#D1C4E9", "#4527A0")   # 残差通道


def _ax(w, h, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ylim)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, pal, fs=9, hatch=None, lw=1.3):
    fc, ec = pal
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.25,rounding_size=0.9",
                                fc=fc, ec=ec, lw=lw, hatch=hatch, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)


def circ(ax, cx, cy, r=1.5, text="$\\oplus$", fs=13):
    ax.add_patch(Circle((cx, cy), r, fc="white", ec="#444444", lw=1.4, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)


def arr(ax, x1, y1, x2, y2, color="#555555", lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def txt(ax, cx, cy, text, fs=8, color="#555555", style="italic"):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color,
            style=style, zorder=3)


def title(ax, text, ylim):
    ax.text(50, ylim - 2.5, text, ha="center", va="center", fontsize=12.5,
            weight="bold", color="#222222")


def caption(ax, text):
    ax.text(50, 1.6, text, ha="center", va="center", fontsize=8, color="#666666")


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ---------------------------------------------------------------- 1. FNO ----
def arch_fno():
    fig, ax = _ax(11, 4.2, 40)
    title(ax, "FNO2d2608 —— 傅里叶神经算子（exp102 基线）", 40)
    box(ax, 7.5, 20, 11, 12, "输入\n18 通道\n65×65\nR,Z+5参数\n+11线圈", GRAY, fs=8)
    box(ax, 21.5, 20, 9, 9, "Lift\n1×1 conv\n18→64", BLUE, fs=8.5)
    box(ax, 44, 20, 28, 27, "", BLUEL, lw=1.6)
    ax.text(44, 31.2, "FNO Block × 4", ha="center", fontsize=9.5, weight="bold")
    box(ax, 37, 25.5, 13.5, 9, "谱路径\nrFFT2 → 线性\n(保留低阶模态)\n→ irFFT2", ORANGE, fs=7.6)
    box(ax, 37, 13.5, 13.5, 5.5, "1×1 conv（逐点）", BLUE, fs=7.8)
    circ(ax, 48.5, 20, 1.5)
    box(ax, 53.5, 20, 6, 4.5, "GELU", BLUE, fs=8)
    arr(ax, 30.4, 20.6, 33.6, 24.2)
    arr(ax, 30.4, 19.4, 33.6, 15.4)
    arr(ax, 43.9, 24.8, 47.2, 21.2)
    arr(ax, 43.9, 14.2, 47.2, 18.8)
    arr(ax, 50.1, 20, 50.4, 20)
    arr(ax, 56.6, 20, 57.6, 20)
    box(ax, 64, 20, 9, 9, "Proj\n1×1 conv\n64→2", BLUE, fs=8.5)
    box(ax, 78, 20, 11, 12, "输出 2ch\nψ_plasma\n+ Jφ\n(z 域)", GRAY, fs=8)
    arr(ax, 13, 20, 17, 20)
    arr(ax, 26, 20, 30, 20)
    arr(ax, 58, 20, 59.5, 20)
    arr(ax, 68.5, 20, 72.5, 20)
    txt(ax, 90.5, 26, "谱卷积 = 全局混合\n（固定傅里叶模态，\n内容无关）", fs=7.6)
    caption(ax, "4.21M 参数 ｜ 4 层 ×（谱卷积 + 1×1），宽度 64，模态 16×16 ｜ exp102 test rel L2 0.800%（N=500, seed 1, data_v5/dn）")
    save(fig, "arch_fno.png")


# --------------------------------------------------------------- 2. UNet ----
def arch_unet():
    fig, ax = _ax(11, 4.4, 42)
    title(ax, "UNet2d2608 —— 纯卷积 U-Net（exp301，失败对照）", 42)
    box(ax, 7, 33.5, 12, 5.5, "输入 18ch 65×65", GRAY, fs=8)
    box(ax, 7, 26.5, 13, 5.5, "Enc1 Conv×2 → 24", BLUE, fs=7.8)
    box(ax, 7, 19.5, 13, 5.5, "Enc2 Conv×2 → 48", BLUE, fs=7.8)
    box(ax, 7, 12.5, 13, 5.5, "Enc3 Conv×2 → 96", BLUE, fs=7.8)
    box(ax, 7, 5.5, 13, 5.5, "Enc4 Conv×2 → 192", BLUE, fs=7.8)
    box(ax, 25, 5.5, 14, 5.5, "Bottleneck 192", BLUE, fs=8)
    box(ax, 43, 12.5, 13, 5.5, "Dec3 Up+Conv → 96", BLUE, fs=7.6)
    box(ax, 43, 19.5, 13, 5.5, "Dec2 Up+Conv → 48", BLUE, fs=7.6)
    box(ax, 43, 26.5, 13, 5.5, "Dec1 Up+Conv → 24", BLUE, fs=7.6)
    box(ax, 43, 33.5, 11, 5.5, "Proj 1×1 → 2", BLUE, fs=7.8)
    box(ax, 60, 33.5, 13, 5.5, "输出 2ch ψ,J", GRAY, fs=8)
    arr(ax, 7, 30.75, 7, 29.25)
    arr(ax, 7, 23.75, 7, 22.25)
    arr(ax, 7, 16.75, 7, 15.25)
    arr(ax, 7, 9.75, 7, 8.25)
    arr(ax, 13.5, 5.5, 18, 5.5)
    arr(ax, 32, 5.8, 36.4, 9.6)
    arr(ax, 43, 15.25, 43, 16.75)   # dec3 -> dec2
    arr(ax, 43, 22.25, 43, 23.75)   # dec2 -> dec1
    arr(ax, 43, 29.25, 43, 30.75)   # dec1 -> proj
    arr(ax, 48.5, 33.5, 53.5, 33.5)
    # skips (straight, middle is empty)
    arr(ax, 13.6, 26.5, 36.4, 26.5, color="#888888", ls="--", lw=1.3)
    arr(ax, 13.6, 19.5, 36.4, 19.5, color="#888888", ls="--", lw=1.3)
    arr(ax, 13.6, 12.5, 36.4, 12.5, color="#888888", ls="--", lw=1.3)
    txt(ax, 25, 29.2, "skip 拼接", fs=7.5)
    txt(ax, 14.6, 23.1, "MaxPool 2×", fs=6.6, style="normal")
    txt(ax, 14.6, 16.1, "MaxPool 2×", fs=6.6, style="normal")
    txt(ax, 14.6, 9.1, "MaxPool 2×", fs=6.6, style="normal")
    txt(ax, 34.6, 15.9, "bilinear 上采样", fs=6.6, style="normal")
    txt(ax, 34.6, 22.9, "bilinear 上采样", fs=6.6, style="normal")
    box(ax, 78, 20, 19, 13,
        "纯局部感受野\n3×3 conv + MaxPool 链\n无谱 / 注意力混合\n——\n2.76M 参数\ntest 2.202%（失败：\nX 点 3-4×、GS 4.3×）",
        ("#FFEBEE", "#B71C1C"), fs=7.4)
    caption(ax, "解码块 = bilinear 上采样 + skip 拼接 + Conv×2，无归一化（与 FNO 基线对齐）｜ 局部感受野不足承载磁面全局拓扑 → 全局性假设的起点")
    save(fig, "arch_unet.png")


# --------------------------------------------------------------- 3. UFNO ----
def arch_ufno():
    fig, ax = _ax(11, 4.4, 42)
    title(ax, "UFNO2d2608 —— 多尺度谱 U-Net（exp302，击败 FNO）", 42)
    box(ax, 7, 33.5, 12, 5.5, "输入 18ch 65×65", GRAY, fs=8)
    box(ax, 7, 26.5, 13, 5.5, "FNO Block 64/16²", ORANGE, fs=7.6)
    box(ax, 7, 19.5, 13, 5.5, "FNO Block 64/16²", ORANGE, fs=7.6)
    box(ax, 7, 12.5, 13, 5.5, "FNO Block 32/8²", ORANGE, fs=7.6)
    box(ax, 7, 5.5, 13, 5.5, "FNO Block 32/4²", ORANGE, fs=7.6)
    box(ax, 25, 5.5, 14, 5.5, "FNO Bottleneck", ORANGE, fs=8)
    box(ax, 43, 12.5, 13, 5.5, "Conv 上采样 → 32", BLUE, fs=7.6)
    box(ax, 43, 19.5, 13, 5.5, "Conv 上采样 → 64", BLUE, fs=7.6)
    box(ax, 43, 26.5, 13, 5.5, "Conv 上采样 → 64", BLUE, fs=7.6)
    box(ax, 43, 33.5, 11, 5.5, "Proj 1×1 → 2", BLUE, fs=7.8)
    box(ax, 60, 33.5, 13, 5.5, "输出 2ch ψ,J", GRAY, fs=8)
    arr(ax, 7, 30.75, 7, 29.25)
    arr(ax, 7, 23.75, 7, 22.25)
    arr(ax, 7, 16.75, 7, 15.25)
    arr(ax, 7, 9.75, 7, 8.25)
    arr(ax, 13.5, 5.5, 18, 5.5)
    arr(ax, 32, 5.8, 36.4, 9.6)
    arr(ax, 43, 15.25, 43, 16.75)
    arr(ax, 43, 22.25, 43, 23.75)
    arr(ax, 43, 29.25, 43, 30.75)
    arr(ax, 48.5, 33.5, 53.5, 33.5)
    arr(ax, 13.6, 26.5, 36.4, 26.5, color="#888888", ls="--", lw=1.3)
    arr(ax, 13.6, 19.5, 36.4, 19.5, color="#888888", ls="--", lw=1.3)
    arr(ax, 13.6, 12.5, 36.4, 12.5, color="#888888", ls="--", lw=1.3)
    txt(ax, 25, 29.2, "skip 拼接", fs=7.5)
    txt(ax, 14.6, 23.1, "MaxPool 2×", fs=6.6, style="normal")
    txt(ax, 14.6, 16.1, "MaxPool 2×", fs=6.6, style="normal")
    txt(ax, 14.6, 9.1, "MaxPool 2×", fs=6.6, style="normal")
    box(ax, 78, 20, 19, 13,
        "编码路径全谱混合\n（多尺度：高分辨率保细节\n+ 低分辨率承载全局）\n解码器纯卷积\n——\n2.47M 参数（FNO 0.59×）\ntest 0.729%（−9%）",
        ("#FFF3E0", "#E65100"), fs=7.4)
    caption(ax, "FNO Block = 谱卷积 + 1×1 + GELU（同 FNO）；模态随分辨率收缩 16²→8²→4² ｜ exp305 谱化解码器微调至 0.7147%")
    save(fig, "arch_ufno.png")


# ------------------------------------------------------------ 4. FNO-KAN ----
def arch_fnokan():
    fig, ax = _ax(11, 4.2, 40)
    title(ax, "FNOKAN2d2608 —— FNO + KAN 混合（exp303，中性）", 40)
    box(ax, 7.5, 20, 11, 12, "输入\n18 通道\n65×65", GRAY, fs=8)
    box(ax, 21.5, 20, 9, 9, "Lift\n1×1 conv\n18→64", BLUE, fs=8.5)
    box(ax, 44, 20, 28, 27, "", BLUEL, lw=1.6)
    ax.text(44, 31.2, "FNO-KAN Block × 4", ha="center", fontsize=9.5, weight="bold")
    box(ax, 37, 25.5, 13.5, 8, "谱卷积\nSpectralConv\n（全局混合）", ORANGE, fs=7.6)
    box(ax, 37, 13.5, 13.5, 9, "KAN 1×1\n逐点 B-spline 样条\n（可学习激活）", TEAL, fs=7.4)
    circ(ax, 48.5, 20, 1.5)
    box(ax, 53.5, 20, 6, 4.5, "GELU", BLUE, fs=8)
    arr(ax, 30.4, 20.6, 33.6, 24.2)
    arr(ax, 30.4, 19.4, 33.6, 15.4)
    arr(ax, 43.9, 24.6, 47.2, 21.2)
    arr(ax, 43.9, 14.4, 47.2, 18.8)
    arr(ax, 50.1, 20, 50.4, 20)
    arr(ax, 56.6, 20, 57.6, 20)
    box(ax, 64, 20, 9, 9, "Proj\n1×1 conv\n64→2", BLUE, fs=8.5)
    box(ax, 78, 20, 11, 12, "输出 2ch\nψ,J", GRAY, fs=8)
    arr(ax, 13, 20, 17, 20)
    arr(ax, 26, 20, 30, 20)
    arr(ax, 58, 20, 59.5, 20)
    arr(ax, 68.5, 20, 72.5, 20)
    txt(ax, 90.5, 26, "KANO 论点：变系数 PDE\n（GS 的 R 因子）\n需逐点可学习激活", fs=7.4)
    caption(ax, "4.33M（+3% vs FNO）｜ test 0.857%（与 FNO 差 ≤0.1pp，机制成立增益零）｜ 代价：训练 ~4× 时长 ｜ 样条域 tanh 门控")
    save(fig, "arch_fnokan.png")


# ----------------------------------------------------------- 5. DeepONet ----
def arch_deeponet():
    fig, ax = _ax(11, 4.2, 40)
    title(ax, "PIDeepONet2d —— 分支-主干点积算子（exp304 / exp307 加宽）", 40)
    box(ax, 11, 28, 17, 9, "16 标量\nIp/paxis/fvac/αm/αn\n+ 11 线圈电流\n（通道空间均值）", GRAY, fs=7.4)
    box(ax, 11, 9, 17, 7, "(R,Z) 网格坐标\n65×65 逐点", GRAY, fs=7.8)
    box(ax, 34, 28, 16, 8, "Branch MLP\n384×3 层 GELU\n（exp307 加宽）", GREEN, fs=7.8)
    box(ax, 34, 9, 16, 8, "Trunk MLP\n384×3 层 GELU\n（末层 ×0.1 初始化）", GREEN, fs=7.8)
    circ(ax, 56, 19, 2.2, text="$\\otimes$", fs=15)
    box(ax, 75, 19, 16, 10, "输出 ψ, J\n逐点内积后\nreshape 65×65", GRAY, fs=8)
    arr(ax, 19.6, 28, 26, 28)
    arr(ax, 19.6, 9, 26, 9)
    arr(ax, 42.2, 27.5, 53.6, 20.9)
    arr(ax, 42.2, 10.5, 53.6, 17.1)
    arr(ax, 58.4, 19, 67, 19)
    txt(ax, 56, 30.5, "ψ(r) = Σ_k  b_k · t_k(r)   （逐点内积 = 算子离散化）", fs=8.2)
    txt(ax, 90.5, 26, "分支-主干分解与任务\n精确同构：\n标量条件 → 场输出", fs=7.4)
    caption(ax, "exp304：0.60M（FNO 的 1/7）→ 0.802%（与 FNO 不可区分），训练最快 ~20 min ｜ exp307 加宽 1.34M → 0.7575%（Ip −45% 反超 UFNO）")
    save(fig, "arch_deeponet.png")


# ------------------------------------------------------------ 6. TKNO ----
def arch_tkno():
    fig, ax = _ax(11.5, 4.2, 40)
    title(ax, "TKNO-lite —— Transformer 全局注意力算子（exp311，系列最优）", 40)
    box(ax, 6.5, 20, 9, 11, "输入\n18ch\n65×65", GRAY, fs=8)
    box(ax, 17, 20, 11, 9, "Conv Stem\n3×3, 18→64\n(65² 细节)", BLUE, fs=7.6)
    box(ax, 31, 20, 12, 9, "Patch Embed\n4×4 / stride 2\n→ 1024 token", YELLOW, fs=7.6)
    circ(ax, 43, 20, 1.3)
    box(ax, 58, 20, 22, 20, "", PURPLEL, lw=1.6)
    ax.text(58, 28.3, "Transformer Block × 4", ha="center", fontsize=9, weight="bold")
    box(ax, 58, 24.5, 17, 5.5, "MHSA（8 头，全局）", PURPLE, fs=7.8)
    box(ax, 58, 14, 17, 5.5, "MLP（扩张 ×4）", PURPLE, fs=7.8)
    txt(ax, 58, 20, "Pre-LN + 残差", fs=7.4)
    box(ax, 76, 20, 10, 9, "Bilinear 上采样\n+ stem skip", YELLOW, fs=7.2)
    box(ax, 90, 20, 9, 11, "输出\n2ch\nψ,J", GRAY, fs=8)
    arr(ax, 11, 20, 11.5, 20)
    arr(ax, 22.5, 20, 25, 20)
    arr(ax, 37, 20, 41.6, 20)
    arr(ax, 44.3, 20, 47, 20)
    arr(ax, 69, 20, 71, 20)
    arr(ax, 81, 20, 85.5, 20)
    txt(ax, 43, 24.5, "＋可学习\n2D 位置编码", fs=6.8)
    # stem skip: elbow under the chain
    ax.plot([17, 17], [15.5, 5.5], color="#888888", lw=1.4, ls="--", zorder=1)
    ax.plot([17, 75.2], [5.5, 5.5], color="#888888", lw=1.4, ls="--", zorder=1)
    arr(ax, 75.2, 5.5, 75.2, 15.4, color="#888888", ls="--", lw=1.4)
    txt(ax, 46, 3.6, "stem skip：65² 细节特征直通解码", fs=7.3)
    caption(ax, "1.21M 参数（FNO 0.29×）｜ test 0.635% 系列最优（J 0.977% / X 点 0.42 cm / O 点 0.18 cm 全最佳）｜ 全局注意力 ≥ 谱混合")
    save(fig, "arch_tkno.png")


# ------------------------------------------------------- 7. POD-DeepONet ----
def arch_pod():
    fig, ax = _ax(11, 4.4, 44)
    title(ax, "PODDeepONet2d —— SVD 低秩基替代学习 trunk（exp312，诚实负面）", 44)
    box(ax, 8, 26, 11, 8, "16 标量\n(z-score)", GRAY, fs=8)
    box(ax, 25, 26, 13, 8, "Branch MLP\n256×3", GREEN, fs=8.2)
    box(ax, 41, 26, 10, 8, "系数 b\nψ:3 + J:10", GREEN, fs=8.2)
    circ(ax, 52.5, 26, 1.6, text="$\\otimes$", fs=13)
    box(ax, 52.5, 36, 17, 6.5, "POD 基 Φ（SVD top-p，冻结）\np_psi=3 / p_j=10", RED, fs=7.4, hatch="//")
    box(ax, 52.5, 16, 13, 6, "均值场 ψ0（冻结）", RED, fs=7.6, hatch="//")
    circ(ax, 66, 26, 1.5)
    box(ax, 80, 26, 14, 9, "ψ = ψ0 + b@Φ\n65×65 重建\n（J 同理）", GRAY, fs=8)
    arr(ax, 13.5, 26, 18.5, 26)
    arr(ax, 31.5, 26, 36, 26)
    arr(ax, 46, 26, 50.8, 26)
    arr(ax, 52.5, 32.75, 52.5, 27.7)
    arr(ax, 54.1, 26, 64.4, 26)
    arr(ax, 59.5, 18.4, 64.7, 24.7)
    arr(ax, 67.5, 26, 73, 26)
    txt(ax, 20, 12, "SVD(500 训练快照, z 域)\n能量判据 99.9% → p 自动确定\n基 = 强正则化（噪声鲁棒的卖）", fs=7.3)
    txt(ax, 90.5, 30, "学习 trunk（exp307）\n≈ 自适应非线性 POD\n→ 超出线性子空间\n0.7pp", fs=7.3)
    caption(ax, "0.28M（系列最小）｜ test 1.508%（exp307 的 2×）：截断的 0.08% 能量恰是 X 点尺度结构 ｜ p_psi=3 = 数据集低秩结构诊断")
    save(fig, "arch_pod_deeponet.png")


# ------------------------------------------------------- 8. POD + 残差 ----
def arch_pod_residual():
    fig, ax = _ax(11.5, 4.4, 44)
    title(ax, "PODResidual2d2608 —— POD 固定基 + 并联残差通道（exp313 FNO / exp314 UNet）", 44)
    box(ax, 7, 22, 9, 13, "输入\n18ch\n65×65", GRAY, fs=8)
    box(ax, 20, 33, 11, 6, "16 标量均值", GRAY, fs=7.8)
    box(ax, 35, 33, 13, 6.5, "Branch MLP 256×3", GREEN, fs=7.8)
    box(ax, 52.5, 33, 12.5, 6.5, "b @ Φ_pod\n（SVD 基 冻结）", RED, fs=7.4, hatch="//")
    box(ax, 38, 10, 30, 10,
        "残差通道（全 18ch 输入）\nexp313：小型 FNO（width 32 / 12²模态 / 4 层）\nexp314：瘦身 UNet（base 16 / depth 4）",
        VIOLET, fs=7.6)
    circ(ax, 70, 22, 1.7)
    box(ax, 83, 22, 10, 10, "输出 2ch\nψ,J", GRAY, fs=8)
    arr(ax, 11.5, 27, 14.5, 31.5)
    arr(ax, 25.5, 33, 28.5, 33)
    arr(ax, 41.5, 33, 46.2, 33)
    arr(ax, 58.9, 31.6, 68.4, 23.6)
    arr(ax, 11.5, 17.5, 23, 11.5)
    arr(ax, 53.2, 11.5, 68.4, 20.4)
    arr(ax, 71.8, 22, 78, 22)
    txt(ax, 38, 3.4, "分阶段冻结协议（--pod-pretrain-epochs 60）：e1–60 只训 branch（残差 proj ×0.1 ≈ 0）→ e61 冻结 branch、残差学 y − ψ_pod（阶段2 物理 ramp 同步爬升）",
        fs=6.9)
    caption(ax, "exp313：870K（FNO 0.21×）→ 0.883%（−41% vs POD，X 点 0.54/0.56 cm 反超 FNO）｜ exp314：1.50M（UNet 残差）→ 1.099% ｜ exp312 诊断的落地修复")
    save(fig, "arch_pod_residual.png")


# ------------------------------------------------------ 9. 两阶段训练 ----
def training_twostage():
    fig, ax = _ax(11, 4.4, 40)
    title(ax, "两阶段 PINO 训练协议（exp102 起，全部架构统一）", 40)
    box(ax, 18, 27, 26, 12,
        "阶段 1：监督预训练（e1..T）\nL = MSE(ψ_z) + w_j·masked MSE(J_z)\n（J 只在 core mask 内）", GREEN, fs=8)
    box(ax, 44, 27, 13, 9, "切换判据\nval rel L2 < 3%\n（或 e300 兜底）", YELLOW, fs=7.6)
    box(ax, 75, 27, 28, 12,
        "阶段 2：物理微调（e T..800）\n+ w_pde·‖Δ*ψ + μ0 R J‖²（自洽）\n+ w_ip·‖Σ J·dA − Ip‖²（约束）", ORANGE, fs=8)
    arr(ax, 31, 27, 37.4, 27)
    arr(ax, 50.5, 27, 61, 27)
    box(ax, 75, 16.5, 28, 7, "物理权重预热（ramp）：w_pde / w_ip\n从 0 线性爬升 30 epochs\n—— 防阶段2 首 epoch 梯度爆炸", YELLOW, fs=7.4)
    arr(ax, 75, 21, 75, 20.2, ls="--", lw=1.2)
    box(ax, 44, 16.5, 17, 6, "早停 patience 75\n（仅阶段2 计数）", GRAY, fs=7.6)
    arr(ax, 44, 22.5, 44, 19.6, ls="--", lw=1.2)
    txt(ax, 50, 9.5,
        "GS 残差：Δ*ψ + μ0·R·Jφ = 0（interior 二阶中心差分，core mask 内，pde_scale = 0.362 Wb/m²）", fs=7.8, style="normal")
    txt(ax, 50, 6.5,
        "coil 分离：网络只学 ψ_plasma；评估 ψ_total = ψ_plasma + Σ I_k·G_k（greens 恒等式 6e-8）｜ Ip 归一 ip_scale = 5.51e5 A", fs=7.8, style="normal")
    caption(ax, "N=500（嵌套子集），seed 1，AdamW 1e-3，batch 16，800 epochs ｜ 18ch = R,Z + 5 参数 + 11 线圈电流（可测量量端到端）")
    save(fig, "training_twostage.png")


if __name__ == "__main__":
    arch_fno()
    arch_unet()
    arch_ufno()
    arch_fnokan()
    arch_deeponet()
    arch_tkno()
    arch_pod()
    arch_pod_residual()
    training_twostage()
    print("all done ->", OUT)
