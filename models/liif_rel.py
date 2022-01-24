import torch
from torch import nn
import torch.nn.functional as F

from models import register
from models.liif import XVFIEncoder, Modulator, ModGenerator
from models.common import VINRDataParallel, Encoder
from dataloaders.utils import make_coord


class LIIF3D(nn.Module):
    # Unfold된 z와 상대좌표, cell크기를 받아서 해당 상대 좌표의 mod param 생성
    def __init__(self, z_dim, num_frames, hidden_node=256, depth=4):
        super(LIIF3D, self).__init__()
        self.num_frames = num_frames
        self.in_f = z_dim * 9 + 4      # feature unfold 3d (*27) / coord concat(+2) / cell size concat(+2)
        self.modulator = Modulator(in_f=self.in_f, hidden_node=hidden_node, depth=depth)

    def forward(self, x, query_coord, cell):
        BF, z_dim, H, W = x.shape
        _, Q, _ = query_coord.shape
        B = int(BF / self.num_frames)

        # 주변 3*3 feature concat
        unfold_feat = F.unfold(x, 3, padding=1).contiguous().view(BF, z_dim*9, H, W)

        vx_lst, vy_lst = [-1, 1], [-1, 1]
        eps_shift = 1e-6

        # Global grid 범위가 (-1, 1)일 때 grid 한 칸의 radius h, w 크기
        rh = 1 / H
        rw = 1 / W

        feat_coord = make_coord(unfold_feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(B, 2, H, W)

        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                # For 문을 돌면서 hr grid 기준 가장 가까이 있는 좌표 4개
                query_coord_ = query_coord.clone()
                query_coord_[:, :, 0] += vx * rh + eps_shift
                query_coord_[:, :, 1] += vy * rw + eps_shift
                query_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_query_coord_ = query_coord_.flip(-1).unsqueeze(1).repeat_interleave(5, dim=0)

                # q_feat: (B, query 수, z_dim * 9) 가장 가까운 lr feature 반환
                q_feat = F.grid_sample(unfold_feat, feat_query_coord_,
                                       mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                # q_coord: (B, query 수, z_dim * 9) 가장 가까운 lr coord 반환
                q_coord = F.grid_sample(feat_coord, query_coord_.flip(-1).unsqueeze(1),
                                        mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                rel_coord = query_coord - q_coord
                rel_coord[:, :, 0] *= H
                rel_coord[:, :, 1] *= W

                rel_coord_for_concat = rel_coord.repeat_interleave(5, dim=0)
                inp = torch.cat([q_feat, rel_coord_for_concat], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= H
                rel_cell[:, :, 1] *= W
                rel_cell_for_concat = rel_cell.repeat_interleave(5, dim=0)
                inp = torch.cat([inp, rel_cell_for_concat], dim=-1)

                mod_params = self.modulator(inp.view(BF * Q, -1), BF, Q)
                preds.append(mod_params)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ratio = (area / tot_area).repeat_interleave(5, dim=0)
            pred = torch.stack(pred, 0)
            cur_mod_params = torch.einsum('dbqh,bq->dbqh', pred, ratio)
            ret = ret + cur_mod_params

        return ret  # (Depth, B * F, HW, hidden)


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
        fx = self.flow_generator(inp_ts, mod_params)
        flow_t_x = fx - ft
        flows = flow_t_x.contiguous().view(B, F, H, W, 2).permute(0, 2, 3, 1, 4)
        warped = self.bwarp(frames, flows).permute(0, 2, 1, 3, 4)
        mask = self.mask_generator(warped)
        visibilities = torch.softmax(mask, 1).permute(0, 2, 3, 1)
        masked = torch.einsum('bcfhw,bhwf->bchw', warped, visibilities)
        return masked, warped, visibilities


@register('liif_rel')
def make_liif_flow(common_opt, specified_opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # nf 64: 1266400 / nf 32: 321920
    encoder = XVFIEncoder(in_c=3, num_frames=common_opt.num_frames, nf=32, z_dim=common_opt.z_dim,
                          n_blocks=specified_opt.encoder_blocks, reduce=False)
    liif = LIIF3D(z_dim=common_opt.z_dim, num_frames=common_opt.num_frames, hidden_node=specified_opt.hidden,
                  depth=specified_opt.depth-1)
    flow_generator = ModGenerator(out_dim=2, w0=specified_opt.w0,
                                  hidden_node=specified_opt.hidden, depth=specified_opt.depth)
    mask_generator = Encoder(in_dim=common_opt.num_frames*3, out_dim=common_opt.num_frames)
    vinr = VINR(encoder, liif, flow_generator, mask_generator, common_opt.num_frames, device)
    model = VINRDataParallel(vinr)
    # frames = torch.rand(2, 3, 5, 32, 32)
    # coord = torch.ones(2, 32 * 32, 2)
    # cell = torch.ones(2, 32 * 32, 2)
    # in_t = torch.rand(2, 5).view(-1, 1)       # (10, 1)
    # rel_t = torch.rand(2, 5).view(-1, 1)      # (10, 1)
    # pred, w, v = model(frames, coord, cell, in_t, rel_t)
    return model
