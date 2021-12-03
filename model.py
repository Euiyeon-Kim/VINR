import torch
import numpy as np
from torch import nn


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
        gamma = x[:, 0::2, :, :]
        beta = x[:, 1::2, :, :]
        return gamma, beta


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


class RGBMapper(nn.Module):
    def __init__(self, in_dim, out_dim, w0=200, hidden_node=256, depth=5):
        super(RGBMapper, self).__init__()
        layers = [SirenLayer(in_f=in_dim, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))
        layers.append(SirenLayer(in_f=hidden_node, out_f=out_dim, is_last=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = torch.clamp(x, min=-1, max=1)
        return x


class VINR(nn.Module):
    def __init__(self, encoder, mapper):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.mapper = mapper

    def forward(self, frames, t):
        gamma, beta = self.encoder(frames)
        assert gamma.shape == beta.shape
        b, z_dim, h, w = gamma.shape
        t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, z_dim, h, w)
        encoded = torch.sin(torch.mul(t, gamma) + beta)
        encoded = encoded.permute(0, 2, 3, 1)
        rgb = self.mapper(encoded).permute(0, 3, 1, 2)
        return rgb


if __name__ == '__main__':
    num_frame = 5
    z_dim = 50

    encoder = Encoder(in_dim=3*num_frame, out_dim=z_dim*2)
    mapper = RGBMapper(in_dim=z_dim, out_dim=3)
    model = VINR(encoder, mapper)
    out = model(torch.rand((4, 3, num_frame, 32, 32)), torch.rand(4, 1))
    print(out.shape)




