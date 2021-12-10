import numpy as np

import torch
from torch import nn


class SinActivation(nn.Module):
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


# t -> emplify with learned fourier feature
class ConLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.layer = nn.Linear(ch_in, ch_out, bias=bias)
        if is_first:
            nn.init.uniform_(self.layer.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.layer.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        return self.layer(x)


class LFF(nn.Module):
    def __init__(self, in_dim, hidden_node):
        super(LFF, self).__init__()
        self.ffm = ConLinear(in_dim, hidden_node, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(0, 0), bias=use_bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(0, 0), bias=use_bias),
           )

    def forward(self, x):
        out = x + self.layers(x)
        return out


class Encoder(nn.Module):
    def __init__(self, in_dim=3, nf=64, out_dim=64, n_blocks=6, use_bias=True):
        super(Encoder, self).__init__()
        self.out_dim = out_dim
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_dim, nf, kernel_size=(7, 7), padding=(0, 0), bias=use_bias),
            nn.ReLU(),
            nn.Conv2d(nf, nf * 2, kernel_size=(3, 3), padding=(1, 1), bias=use_bias),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(),
            nn.Conv2d(nf * 2, nf * 4, kernel_size=(3, 3), padding=(1, 1), bias=use_bias),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(),
        ]

        for _ in range(n_blocks):
            layers.append(ResnetBlock(dim=nf * 4, use_bias=use_bias))

        layers.extend(
            [
                nn.Conv2d(nf * 4, nf * 4, kernel_size=(3, 3), padding=(1, 1), bias=use_bias),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU(),
                nn.Conv2d(nf * 4, out_dim, kernel_size=(3, 3), padding=(1, 1), bias=use_bias),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.contiguous().view(b, c * t, h, w)
        x = self.layers(x)
        # gamma = x[:, 0::2, :, :]
        # beta = x[:, 1::2, :, :]
        return x


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super(SirenLayer, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.w0 = w0

        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last

        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            nn.init.uniform_(self.linear.weight, a=-b, b=b)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
