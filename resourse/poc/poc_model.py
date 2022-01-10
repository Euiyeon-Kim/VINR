import torch
from torch import nn
from models.common import SirenLayer


class Reflector(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim, w0=200, hidden_node=256, depth=5):
        super(Reflector, self).__init__()
        self.z_dim = z_dim
        self.out_dim = out_dim
        layers = [SirenLayer(in_f=in_dim, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))
        layers.append(SirenLayer(in_f=hidden_node, out_f=out_dim*z_dim, is_last=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1)
        x = self.layers(x)
        b, _ = x.shape
        x = x.view((b, self.out_dim, self.z_dim))
        return x


class VINR(nn.Module):
    def __init__(self, encoder, mapper):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.mapper = mapper

    def forward(self, frames, t):
        z = self.encoder(frames)
        reflet_t = self.mapper(t)
        rgb = torch.einsum('bzhw,bcz->bchw', z, reflet_t)
        rgb = torch.clamp(rgb, min=-1, max=1)
        return rgb


if __name__ == '__main__':
    from config import Config
    opt = Config()
    from models.common import Encoder
    encoder = Encoder(in_dim=3 * opt.num_frames, out_dim=opt.z_dim)
    mapper = Reflector(in_dim=1, out_dim=3, z_dim=opt.z_dim)
    model = VINR(encoder, mapper)
    out = model(torch.rand((4, 3, opt.num_frames, 32, 32)), torch.rand(4))
    print(out.shape)




