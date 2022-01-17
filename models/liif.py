import torch
from torch import nn
import torch.nn.functional as F

from models import register
from models.common import LFF, SirenLayer, VINRDataParallel
from dataloaders.utils import make_coord

class ResBlock3D(nn.Module):
    def __init__(self, nf):
        super(ResBlock3D, self).__init__()
        self.conv3x3_1 = nn.Conv3d(nf, nf, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.conv3x3_2 = nn.Conv3d(nf, nf, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.lrelu = nn.ReLU()

    def forward(self, x):
        out = self.conv3x3_2(self.lrelu(self.conv3x3_1(x)))
        return x + out


class RResBlock3D(nn.Module):
    def __init__(self, nf):
        super(RResBlock3D, self).__init__()
        self.nf = nf
        self.resblock1 = ResBlock3D(nf)
        self.resblock2 = ResBlock3D(nf)

    def forward(self, x):
        out = self.resblock1(x)
        out = self.resblock2(out)
        return out + x


class XVFIEncoder(nn.Module):
    def __init__(self, in_c=3, num_frames=5, nf=64, n_blocks=2):
        super(XVFIEncoder, self).__init__()
        self.channel_converter = nn.Sequential(
            nn.Conv3d(in_c, nf, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.ReLU()
        )
        self.rec_ext_ds_module = [self.channel_converter]
        self.rec_ext_ds = nn.Conv3d(nf, nf, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        for _ in range(n_blocks):
            self.rec_ext_ds_module.append(self.rec_ext_ds)
            self.rec_ext_ds_module.append(nn.ReLU())
        self.rec_ext_ds_module.append(nn.Conv3d(nf, nf, (1, 3, 3), (1, 1, 1), (0, 1, 1)))
        self.rec_ext_ds_module.append(RResBlock3D(nf))
        self.rec_ext_ds_module = nn.Sequential(*self.rec_ext_ds_module)
        self.reduceT_conv = nn.Conv3d(nf, nf, (num_frames, 1, 1), (1, 1, 1), (0, 0, 0))

        self.nf = nf
        self.conv_last = nn.Sequential(
            nn.Conv2d(nf, 2 * nf, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(2 * nf, 4 * nf, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(4 * nf, 2 * nf, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(2 * nf, nf, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, x):
        feat_x = self.rec_ext_ds_module(x)
        B, z_dim, F, H, W = feat_x.shape
        feat_x = feat_x.permute(0, 2, 1, 3, 4).contiguous().view(B*F, z_dim, H, W)
        feat_x = self.conv_last(feat_x)
        feat_x = feat_x.view(B, F, z_dim, H, W).permute(0, 2, 1, 3, 4)
        feat_x = torch.squeeze(self.reduceT_conv(feat_x), 2)
        return feat_x


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

    def forward(self, z, b, q):
        x = z
        alphas = []
        for layer in self.layers:
            x = layer(x)
            alpha = x.view(b, q, self.hidden_node)
            alphas.append(alpha)
            x = torch.cat((x, z), dim=-1)
        return alphas


class LIIF(nn.Module):
    # Unfold된 z와 상대좌표, cell크기 를 받아서 해당 상대 좌표의 mod param 생성
    def __init__(self, z_dim, hidden_node=256, depth=4):
        super(LIIF, self).__init__()
        self.in_f = z_dim * 9 + 4      # feature unfold(*9) / coord concat(+2) / cell size concat(+2)
        self.modulator = Modulator(in_f=self.in_f, hidden_node=hidden_node, depth=depth)

    def forward(self, x, query_coord, cell):
        # x(feature): (B, z_dim, H, W)
        # coord: (B, sample, 2)
        B, z_dim, H, W = x.shape

        # 주변 3*3 feature concat
        unfold_feat = F.unfold(x, 3, padding=1).view(B, z_dim * 9, H, W)

        vx_lst, vy_lst = [-1, 1], [-1, 1]
        eps_shift = 1e-6

        # Global grid 범위가 (-1, 1)일 때 grid 한 칸의 radius h, w 크기
        rh = 1 / H
        rw = 1 / W

        feat_coord = make_coord(unfold_feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).\
            unsqueeze(0).expand(B, 2, H, W)

        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                # For 문을 돌면서 hr grid 기준 가장 가까이 있는 좌표 4개
                query_coord_ = query_coord.clone()
                query_coord_[:, :, 0] += vx * rh + eps_shift
                query_coord_[:, :, 1] += vy * rw + eps_shift
                query_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # q_feat: (B, query 수, z_dim * 9) 가장 가까운 lr feature 반환
                q_feat = F.grid_sample(unfold_feat, query_coord_.flip(-1).unsqueeze(1),
                                       mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                # q_coord: (B, query 수, z_dim * 9) 가장 가까운 lr coord 반환
                q_coord = F.grid_sample(feat_coord, query_coord_.flip(-1).unsqueeze(1),
                                        mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                rel_coord = query_coord - q_coord
                rel_coord[:, :, 0] *= H
                rel_coord[:, :, 1] *= W
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= H
                rel_cell[:, :, 1] *= W
                inp = torch.cat([inp, rel_cell], dim=-1)

                B, Q = query_coord.shape[:2]
                mod_params = self.modulator(inp.view(B * Q, -1), B, Q)

                preds.append(mod_params)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ratio = (area / tot_area)
            pred = torch.stack(pred, 0)
            cur_mod_params = torch.einsum('dbqh,bq->dbqh', pred, ratio)
            ret = ret + cur_mod_params
        return ret


class ModGenerator(nn.Module):
    def __init__(self, out_dim=3, w0=200, hidden_node=256, depth=5):
        super(ModGenerator, self).__init__()
        self.lff = LFF(1, hidden_node)
        self.depth = depth

        layers = [SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))

        self.layers = nn.Sequential(*layers)
        self.last_layer = SirenLayer(in_f=hidden_node, out_f=out_dim, is_last=True)

    def forward(self, t, mod_params):
        l, b, q, hidden = mod_params.shape
        x = self.lff(t).unsqueeze(1).repeat(1, q, 1)
        for layer, mod in zip(self.layers, mod_params):
            x = layer(x)
            x *= mod
        x = self.last_layer(x)
        x = torch.clamp(x, min=-1, max=1)
        return x


class VINR(nn.Module):
    def __init__(self, encoder, liif, mapper):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.liif = liif
        self.mapper = mapper

    # Normalize?
    def forward(self, frames, query_coord, cell, t):
        feat = self.get_feat(frames)
        pred = self.get_rgb(feat, query_coord, cell, t)
        return pred

    def get_feat(self, frames):     # can only be called by validation or test
        z = self.encoder(frames)
        return z

    def get_rgb(self, feat, query_coord, cell, t):
        mod_params = self.liif(feat, query_coord, cell)
        pred = self.mapper(t, mod_params)
        return pred


@register('liif')
def make_liif(common_opt, specified_opt):
    encoder = XVFIEncoder(in_c=3, num_frames=common_opt.num_frames, nf=common_opt.z_dim,
                          n_blocks=specified_opt.encoder_blocks)
    liif = LIIF(z_dim=common_opt.z_dim, hidden_node=specified_opt.hidden, depth=specified_opt.depth-1)
    mapper = ModGenerator(out_dim=3, w0=specified_opt.w0,
                          hidden_node=specified_opt.hidden, depth=specified_opt.depth)
    vinr = VINR(encoder, liif, mapper)
    model = VINRDataParallel(vinr)
    return model


if __name__ == '__main__':
    batch_size = 7
    num_frames = 5
    z_dim = 50
    h, w = 96, 96
    encoder = XVFIEncoder(in_c=3, num_frames=num_frames, nf=z_dim, n_blocks=2)
    z = encoder(torch.randn((batch_size, 3, num_frames, h, w)))

    cell = torch.randn((batch_size, h*w, 2))
    liif = LIIF(z_dim)
    coord = make_coord([h, w])
    coord = torch.unsqueeze(coord, 0).repeat(batch_size, 1, 1)
    mod_params = liif(z, coord, cell)

    mapper = ModGenerator(out_dim=3)
    rgb = mapper(torch.rand(batch_size, 1), mod_params)
    print(coord.shape, cell.shape)
    model = VINR(encoder, liif, mapper)
    out = model(torch.rand((batch_size, 3, num_frames, h, w)), coord, cell, torch.rand(batch_size, 1))
    print(out.shape)
