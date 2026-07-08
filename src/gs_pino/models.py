"""Neural-operator model definitions used by the GS_PINO workflow.

Contains:
1. PlaNetCore: Trunk-Branch-Decoder architecture following PlaNet-equil
2. UFNO2d: Original U-FNO architecture (kept for reference)
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TrainableSwish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.sigmoid(self.beta * x)


class Conv2dNornAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3), padding: str = "same"):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.act = TrainableSwish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(self.norm(x))
        return x.permute(0, 3, 1, 2)


class TrunkNet(nn.Module):
    def __init__(self, hidden_dim: int = 128, nr: int = 64, nz: int = 64):
        super().__init__()
        assert nr % 2 == 0, f"nr must be a power of 2, got {nr}"
        assert nz % 2 == 0, f"nz must be a power of 2, got {nz}"
        
        self.conv_layers = nn.ModuleList()
        channels = [2, 16, 32, 64, 128]
        for i in range(len(channels) - 1):
            in_channels, out_channels = channels[i], channels[i + 1]
            self.conv_layers.append(
                nn.Sequential(
                    Conv2dNornAct(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
        
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(
            in_features=int(channels[-1] * nr / 2**4 * nz / 2**4), out_features=256
        )
        self.norm_1 = nn.LayerNorm(256)
        self.act_1 = TrainableSwish()
        self.linear_2 = nn.Linear(in_features=256, out_features=hidden_dim)

    def forward(self, x_r: torch.Tensor, x_z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_r.unsqueeze(1), x_z.unsqueeze(1)], dim=1)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.act_1(self.norm_1(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class BranchNet(nn.Module):
    def __init__(self, in_dim: int = 9, hidden_dim: int = 128):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_dim, out_features=256)
        self.norm_1 = nn.LayerNorm(256)
        self.act_1 = TrainableSwish(beta=1.0)
        self.linear_2 = nn.Linear(in_features=256, out_features=512)
        self.norm_2 = nn.LayerNorm(512)
        self.act_2 = TrainableSwish(beta=1.0)
        self.linear_3 = nn.Linear(in_features=512, out_features=256)
        self.norm_3 = nn.LayerNorm(256)
        self.act_3 = TrainableSwish(beta=1.0)
        self.linear_4 = nn.Linear(in_features=256, out_features=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_1(self.norm_1(self.linear_1(x)))
        x = self.act_2(self.norm_2(self.linear_2(x)))
        x = self.act_3(self.norm_3(self.linear_3(x)))
        x = self.linear_4(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, hidden_dim: int = 128, nr: int = 64, nz: int = 64):
        super().__init__()
        assert nr % 2 == 0, f"nr must be a power of 2, got {nr}"
        assert nz % 2 == 0, f"nz must be a power of 2, got {nz}"
        self.nr = nr
        self.nz = nz
        
        self.linear = nn.Linear(
            in_features=hidden_dim,
            out_features=256 * int(self.nr / 2**4) * int(self.nz / 2**4),
        )
        self.act = TrainableSwish()
        self.norm = nn.LayerNorm(256)
        
        self.decoder = nn.ModuleList()
        channels = [256, 128, 64, 32, 16]
        for i in range(len(channels) - 1):
            in_channels, out_channels = channels[i], channels[i + 1]
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                    Conv2dNornAct(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3)),
                )
            )
        
        self.conv_out = nn.Conv2d(in_channels=channels[-1], out_channels=1, kernel_size=(3, 3), padding="same")

    def forward(self, x_trunk: torch.Tensor, x_branch: torch.Tensor) -> torch.Tensor:
        batch = x_branch.shape[0]
        x = x_branch * x_trunk
        x = self.act(self.linear(x))
        x = x.reshape((batch, 256, int(self.nr / 2**4), int(self.nz / 2**4)))
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        for layer in self.decoder:
            x = layer(x)

        x = self.conv_out(x).squeeze(1)
        return x


class PlaNetCore(nn.Module):
    def __init__(self, n_measures: int = 9, hidden_dim: int = 128, nr: int = 64, nz: int = 64):
        super().__init__()
        self.trunk = TrunkNet(hidden_dim=hidden_dim, nr=nr, nz=nz)
        self.branch = BranchNet(in_dim=n_measures, hidden_dim=hidden_dim)
        self.decoder = DecoderConv(hidden_dim=hidden_dim, nr=nr, nz=nz)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_meas, x_r, x_z = x
        out_branch = self.branch(x_meas)
        out_trunk = self.trunk(x_r, x_z)
        return self.decoder(out_branch, out_trunk)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch, self.out_channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)

        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(height, width))


class UNetBranch(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.avg_pool2d(x, 2, ceil_mode=True)
        y = self.down(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.up(torch.cat([x, y], dim=1))


class UFNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.unet = UNetBranch(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.pointwise(x) + self.unet(x))


class UFNO2d(nn.Module):
    def __init__(self, in_channels: int, modes1: int = 16, modes2: int = 16, width: int = 32, layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([UFNOBlock(width, modes1, modes2) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)