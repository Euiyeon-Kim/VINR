import torch
from torch import nn
from common_model import LFF, SirenLayer


class ModFlowMapper(nn.Module):
    def __init__(self, out_dim=3, w0=200, hidden_node=256, depth=5):
        super(ModFlowMapper, self).__init__()
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
    def __init__(self, encoder, modulator, flow_generator):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.modulator = modulator
        self.flow_generator = flow_generator

    def forward(self, frames, t):
        encoded = self.encoder(frames)

        min_v = torch.min(encoded, dim=1, keepdim=True)[0]
        max_v = torch.max(encoded, dim=1, keepdim=True)[0]
        normalized = (((encoded - min_v) / (max_v - min_v)) - 0.5) * 2.0    # (-1, 1)

        mod_params = self.modulator(normalized)
        flow = self.flow_generator(t.unsqueeze(-1), mod_params)
        return flow


if __name__ == '__main__':
    num_frame = 2
    z_dim = 50

    from common_model import Encoder
    from mod_model import Modulator
    encoder = Encoder(in_dim=3*num_frame, out_dim=z_dim)
    modulator = Modulator(z_dim, 256, 4)

    flow_generator = ModFlowMapper(out_dim=2)
    model = VINR(encoder, modulator, flow_generator)

    out = model(torch.rand((4, 3, num_frame, 32, 32)), torch.rand(4))
    print(out.shape)




