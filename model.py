from typing import Tuple

import torch
from torch import nn, Tensor

from normalized_vgg import Vgg19, Vgg19Features


class ImageTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Vgg19(use_avg_pooling=True)
        self.decoder = Decoder()

    def forward(self, content: Tensor, style: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor, Vgg19Features]:
        with torch.no_grad():
            content_enc = self.encoder(content)
            style_enc = self.encoder(style)

        content_norm = self._adaptive_instance_norm(content_enc.relu4_1, style_enc.relu4_1)
        content_norm = alpha * content_norm + (1 - alpha) * content_enc.relu4_1
        stylized = self.decoder(content_norm)
        return stylized, content_norm, style_enc

    def encode(self, x: Tensor) -> Vgg19Features:
        return self.encoder(x)

    @staticmethod
    def _adaptive_instance_norm(x: Tensor, y: Tensor) -> Tensor:
        x_mu, x_sigma = calc_stats(x)
        y_mu, y_sigma = calc_stats(y)
        eps = torch.tensor(1e-10, dtype=torch.float32, device=x.device).expand_as(x_sigma)
        x = (x - x_mu) / torch.maximum(x_sigma, eps)
        x = x * y_sigma + y_mu
        return x


class Decoder(nn.Sequential):
    def __init__(self):
        layers = (
            nn.Conv2d(512, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )

        super().__init__(*layers)


def calc_stats(x: Tensor) -> Tuple[Tensor, Tensor]:
    mu = x.mean(dim=(2, 3), keepdim=True)
    sigma = x.std(dim=(2, 3), keepdim=True)
    return mu, sigma
