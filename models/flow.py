import torch
from torch import nn
from models.common import LFF, SirenLayer


class ModMapper(nn.Module):
    def __init__(self, out_dim=3, w0=200, hidden_node=256, depth=5):
        super(ModMapper, self).__init__()
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
        print(x.shape)
        for layer, mod in zip(self.layers, mod_params):
            x = layer(x)
            x *= mod
        x = self.last_layer(x)
        return x


class VINR(nn.Module):
    def __init__(self, encoder, modulator, mod_generator, device, num_frame):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.modulator = modulator
        self.mod_generator = mod_generator
        self.device = device
        self.num_frame = num_frame

    def bwarp(self, frames, flows):
        """
        :param frames: B, C, F, H, W
        :param flows: B, H, W, F * 2
        :return: warped frames
        """
        B, C, F, H, W = frames.shape
        flows = flows.view(B, H, W, F, 2).permute(0, 3, 1, 2, 4).contiguous().view(B*F, H, W, 2)
        frames = frames.permute(0, 2, 1, 3, 4).contiguous().view(B*F, C, H, W)

        xx = torch.arange(0, W).view(1, 1, W, 1).expand(B*F, H, W, 1)
        yy = torch.arange(0, H).view(1, H, 1, 1).expand(B*F, H, W, 1)
        grid = torch.cat((xx, yy), -1).float().to(self.device)

        vgrid = torch.autograd.Variable(grid) + flows
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0].clone() / max(W - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1].clone() / max(H - 1, 1) - 1.0

        warped = nn.functional.grid_sample(frames, vgrid, align_corners=True)
        warped = warped.view(B, F, C, H, W)
        return warped

    def forward(self, frames, t):
        encoded = self.encoder(frames)

        min_v = torch.min(encoded, dim=1, keepdim=True)[0]
        max_v = torch.max(encoded, dim=1, keepdim=True)[0]
        normalized = (((encoded - min_v) / (max_v - min_v)) - 0.5) * 2.0    # (-1, 1)

        mod_params = self.modulator(normalized)
        flows_and_visibilities = self.mod_generator(t.unsqueeze(-1), mod_params)

        flows = flows_and_visibilities[:, :, :, :self.num_frame*2]
        visibilities = torch.softmax(flows_and_visibilities[:, :, :, self.num_frame*2:], -1)

        # warped = self.bwarp(frames, flows)
        # masked = torch.einsum('bfchw,bhwf->bchw', warped, visibilities)

        warped = self.bwarp(frames, flows).permute(0, 2, 1, 3, 4)
        masked = torch.einsum('bcfhw,bhwf->bchw', warped, visibilities)

        return masked, warped, visibilities
