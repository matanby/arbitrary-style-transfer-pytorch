from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class Vgg19Features:
    relu1_1: Tensor
    relu2_1: Tensor
    relu3_1: Tensor
    relu4_1: Tensor
    relu5_1: Tensor


class Vgg19(nn.Module):
    def __init__(self, use_avg_pooling: bool = False):
        super(Vgg19, self).__init__()

        pool = nn.AvgPool2d if use_avg_pooling else nn.MaxPool2d
        self._model = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1_2
            pool((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2_2
            pool((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3_2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3_3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3_4
            pool((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4_2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4_3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4_4
            pool((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5_1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5_2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5_3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5_4
        )

        weights = torch.load('vgg_normalised.pth')
        self._model.load_state_dict(weights)

        self.block_1 = nn.Sequential(*self._model[:4])  # relu1_1
        self.block_2 = nn.Sequential(*self._model[4:11])  # relu2_1
        self.block_3 = nn.Sequential(*self._model[11:18])  # relu3_1
        self.block_4 = nn.Sequential(*self._model[18:31])  # relu4_1
        self.block_5 = nn.Sequential(*self._model[31:43])  # relu5_1

        # TODO: remove?
        mean = Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._mean = nn.Parameter(mean, requires_grad=False)
        self._std = nn.Parameter(std, requires_grad=False)

        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, img: Tensor) -> Vgg19Features:
        # TODO: remove?
        # img = (img - self._mean) / self._std
        h_relu1 = self.block_1(img)
        h_relu2 = self.block_2(h_relu1)
        h_relu3 = self.block_3(h_relu2)
        h_relu4 = self.block_4(h_relu3)
        h_relu5 = self.block_5(h_relu4)
        out = Vgg19Features(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out
