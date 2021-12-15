import torch
from torch import nn


class LIIF(nn.Module):
    def __init__(self, in_f, hidden_node, depth):
        super(LIIF, self).__init__()
        self.in_f = in_f
        self.hidden_node = hidden_node

        layers = []
        for i in range(depth):
            dim = in_f if i == 0 else (hidden_node + in_f)
            layers.append(nn.Sequential(
                nn.Linear(dim, hidden_node),
                nn.ReLU()
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class VINR(nn.Module):
    def __init__(self, encoder, liif, modulator, mapper):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.liif = liif
        self.modulator = modulator
        self.mapper = mapper

    def forward(self, frames, t):
        encoded = self.encoder(frames)
        continuous_feature = self.liif(encoded)

        # TODO: Namalization?

        mod_params = self.modulator(continuous_feature)
        rgb = self.mapper(t.unsqueeze(-1), mod_params).permute(0, 3, 1, 2)
        return rgb


if __name__ == '__main__':
    num_frame = 5
    z_dim = 50

    from common_model import Encoder
    from mod_model import Modulator, ModRGBMapper

    encoder = Encoder(in_dim=3*num_frame, out_dim=z_dim)
    z = encoder(torch.randn((4, 3, 5, 32, 32)))

    liif = LIIF()
    continuous_feature = liif(z)

    modulator = Modulator(continuous_feature, 256, 4)
    mod_params = modulator(z)

    mapper = ModRGBMapper(out_dim=3)
    rgb = mapper(torch.rand(4, 1), mod_params)

    model = VINR(encoder, modulator, mapper)
    out = model(torch.rand((4, 3, num_frame, 32, 32)), torch.rand(4))
    print(out.shape)
