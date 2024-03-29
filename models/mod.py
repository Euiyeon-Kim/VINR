import torch
from torch import nn

from models import register
from models.common import LFF, SirenLayer, Encoder, VINRDataParallel


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
        x = self.lff(t).unsqueeze(1).repeat(1, h, w, 1)
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
        feat = self.get_feat(frames)
        rgb = self.get_rgb(feat, t)
        return rgb

    def get_feat(self, frames):
        encoded = self.encoder(frames)
        min_v = torch.min(encoded, dim=1, keepdim=True)[0]
        max_v = torch.max(encoded, dim=1, keepdim=True)[0]
        normalized = (((encoded - min_v) / (max_v - min_v)) - 0.5) * 2.0    # (-1, 1)
        return normalized

    def get_rgb(self, normalized, t):
        mod_params = self.modulator(normalized)
        rgb = self.mapper(t.unsqueeze(-1), mod_params).permute(0, 3, 1, 2)
        return rgb



@register('mod')
def make_mod_vinr(common_opt, specified_opt):
    encoder = Encoder(in_dim=3*common_opt.num_frames, out_dim=common_opt.z_dim)
    modulator = Modulator(common_opt.z_dim, specified_opt.hidden, specified_opt.depth-1)
    mapper = ModRGBMapper(out_dim=3, w0=specified_opt.w0, hidden_node=specified_opt.hidden, depth=specified_opt.depth)
    vinr = VINR(encoder, modulator, mapper)
    model = VINRDataParallel(vinr)
    return model





