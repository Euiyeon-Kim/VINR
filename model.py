import torch
import numpy as np
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


class Modulator(nn.Module):
    def __init__(self, in_f, hidden_node=256, depth=4):
        super(Modulator, self).__init__()
        self.hidden_node = hidden_node
        self.layers = nn.ModuleList([])

        for i in range(depth):
            dim = in_f if i == 0 else (hidden_node + in_f)
            self.layers.append(nn.Sequential(
                nn.Linear(dim, hidden_node),
                nn.ReLU()
            ))

    def forward(self, z):
        b, z_dim, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(-1, z_dim)
        x = z
        alphas = []
        for layer in self.layers:
            x = layer(x)
            alpha = x.view(b, h, w, self.hidden_node)
            alphas.append(alpha)
            x = torch.cat((x, z), dim=-1)
        return tuple(alphas)


class ModRGBMapper(nn.Module):
    def __init__(self, out_dim=3, w0=200, hidden_node=256, depth=5):
        super(ModRGBMapper, self).__init__()
        self.lff = LFF(1, hidden_node)
        self.depth = depth

        layers = [SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))

        self.layers = nn.Sequential(*layers)
        self.last_layer = SirenLayer(in_f=hidden_node, out_f=out_dim, is_last=True)

    def forward(self, t, mod_params):
        b, h, w, hidden = mod_params[0].shape
        x = self.lff(t).unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1)
        for layer, mod in zip(self.layers, mod_params):
            x = layer(x)
            x *= mod
        x = self.last_layer(x)
        x = torch.clamp(x, min=-1, max=1)
        return x


class VINR(nn.Module):
    def __init__(self, encoder, modulator, mapper):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.modulator = modulator
        self.mapper = mapper

    def forward(self, frames, t):
        encoded = self.encoder(frames)
        encoded = torch.nn.functional.normalize(encoded, p=1.0)
        mod_params = self.modulator(encoded)
        rgb = self.mapper(t.unsqueeze(-1), mod_params).permute(0, 3, 1, 2)
        return rgb


if __name__ == '__main__':
    num_frame = 5
    z_dim = 50

    encoder = Encoder(in_dim=3*num_frame, out_dim=z_dim)
    z = encoder(torch.randn((4, 3, 5, 32, 32)))

    modulator = Modulator(z_dim, 256, 4)
    mod_params = modulator(z)

    mapper = ModRGBMapper(out_dim=3)
    rgb = mapper(torch.rand(4, 1), mod_params)

    model = VINR(encoder, modulator, mapper)
    out = model(torch.rand((4, 3, num_frame, 32, 32)), torch.rand(4))
    print(out.shape)




