import torch
from torch import nn
import torch.nn.functional as F

from models import register
from models.liif import XVFIEncoder, LIIF, ModGenerator
from models.common import VINRDataParallel


class VINR(nn.Module):
    def __init__(self, encoder, liif, flow_generator, mask_generator, num_frames, device):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.liif = liif
        self.flow_generator = flow_generator
        self.mask_generator = mask_generator
        self.num_frames = num_frames
        self.device = device

    def bwarp(self, frames, flows):
        """
        :param frames: B, C, F, H, W
        :param flows: B, H, W, F, 2
        :return: warped frames
        """
        B, C, F, H, W = frames.shape
        flows = flows.permute(0, 3, 1, 2, 4).contiguous().view(B*F, H, W, 2)
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

    # Normalize?
    def forward(self, frames, query_coord, cell, inp_ts, t):
        feat = self.get_feat(frames)
        pred = self.get_rgb(frames, feat, query_coord, cell, inp_ts, t)
        return pred

    def get_feat(self, frames):
        z = self.encoder(frames)
        return z

    def get_rgb(self, frames, feat, query_coord, cell, inp_ts, t):
        B, C, F, H, W = frames.shape

        mod_params = self.liif(feat, query_coord, cell)

        ft = self.flow_generator(t, mod_params)

        flows = []
        for inp_t in inp_ts.permute(1, 0):
            fx = self.flow_generator(torch.unsqueeze(inp_t, -1), mod_params)
            flow_t_x = fx - ft
            flows.append(flow_t_x)
        flows = torch.stack(flows, dim=-2).contiguous().view(-1, H, W, F, 2)

        warped = self.bwarp(frames, flows).permute(0, 2, 1, 3, 4)
        visibilities = torch.softmax(self.mask_generator(t, mod_params), -1).contiguous().view(-1, H, W, F)
        masked = torch.einsum('bcfhw,bhwf->bchw', warped, visibilities)
        return masked, warped, visibilities


@register('liif_flow_v2')
def make_liif_flow(common_opt, specified_opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = XVFIEncoder(in_c=3, num_frames=common_opt.num_frames, z_dim=common_opt.z_dim,
                          n_blocks=specified_opt.encoder_blocks)
    liif = LIIF(z_dim=common_opt.z_dim, hidden_node=specified_opt.hidden, depth=specified_opt.depth-1)
    flow_generator = ModGenerator(out_dim=2, w0=specified_opt.w0,
                                  hidden_node=specified_opt.hidden, depth=specified_opt.depth)
    mask_generator = ModGenerator(out_dim=common_opt.num_frames, w0=specified_opt.w0,
                                  hidden_node=specified_opt.hidden, depth=specified_opt.depth)
    vinr = VINR(encoder, liif, flow_generator, mask_generator, common_opt.num_frames, device)
    model = VINRDataParallel(vinr)
    return model
