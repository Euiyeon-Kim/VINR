import torch
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
    def __init__(self, n_blocks=6):
        super(Encoder, self).__init__()
        layers = []


    def forward(self, x):
        pass


if __name__ == '__main__':
    encoder = Encoder()
    print(encoder)



