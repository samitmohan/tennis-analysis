"""TrackNetV2-style ball heatmap network (Sun et al. 2020).

Three consecutive frames are stacked on the channel axis (9 x H x W). A VGG-style
U-Net returns one heatmap of logits for the last frame. One frame shows a small,
blurred ball; three frames show its motion, which is what separates it from a
white shoe or a logo.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _block(cin: int, cout: int, n: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(n):
        layers += [
            nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cout),
        ]
    return nn.Sequential(*layers)


class TrackNet(nn.Module):
    def __init__(self, frames: int = 3, out_ch: int = 1) -> None:
        super().__init__()
        self.e1 = _block(3 * frames, 64, 2)
        self.e2 = _block(64, 128, 2)
        self.e3 = _block(128, 256, 3)
        self.e4 = _block(256, 512, 3)
        self.d3 = _block(512 + 256, 256, 3)
        self.d2 = _block(256 + 128, 128, 2)
        self.d1 = _block(128 + 64, 64, 2)
        self.head = nn.Conv2d(64, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        y = self.d3(torch.cat([self.up(x4), x3], 1))
        y = self.d2(torch.cat([self.up(y), x2], 1))
        y = self.d1(torch.cat([self.up(y), x1], 1))
        return self.head(y)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 100.0) -> torch.Tensor:
    """BCE with the ~29 ball pixels weighted 100x against ~147k background pixels.

    A focal variant (p^2 on negatives) was tried first and never learned: on 64 frames
    for 300 steps it decoded 0/32 balls, with every pixel parked near 0.13. The p^2
    factor makes the background gradient vanish at low p. This loss decodes 32/32
    within 5 px on the same test (fp16 and fp32).
    """
    w = torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=w)
