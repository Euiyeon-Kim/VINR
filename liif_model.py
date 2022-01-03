import torch
from torch import nn
import torch.nn.functional as F

from common_model import LFF, SirenLayer


class ResBlock3D(nn.Module):
    def __init__(self, nf):
        super(ResBlock3D, self).__init__()
        self.conv3x3_1 = nn.Conv3d(nf, nf, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.conv3x3_2 = nn.Conv3d(nf, nf, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.lrelu = nn.ReLU()

    def forward(self, x):
        B, C, F, H, W = x.size()
        out = self.conv3x3_2(self.lrelu(self.conv3x3_1(x)))
        return x + out


class RResBlock3D(nn.Module):
    def __init__(self, nf, num_frames, reduce_f=False):
        super(RResBlock3D, self).__init__()
        self.nf = nf
        self.reduce_f = reduce_f
        self.resblock1 = ResBlock3D(nf)
        self.resblock2 = ResBlock3D(nf)
        self.reduceT_conv = nn.Conv3d(nf, nf, (num_frames, 1, 1), (1, 1, 1), (0, 0, 0))

    def forward(self, x):
        out = self.resblock1(x)
        out = self.resblock2(out)
        return torch.squeeze(self.reduceT_conv(out + x))


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
        self.rec_ext_ds_module.append(RResBlock3D(nf, num_frames=num_frames, reduce_f=False))
        self.rec_ext_ds_module = nn.Sequential(*self.rec_ext_ds_module)

    def forward(self, x):
        feat_x = self.rec_ext_ds_module(x)
        return feat_x


class LIIF(nn.Module):
    # Unfold된 z와 상대좌표, cell크기 를 받아서 해당 상대 좌표의 mod param 생성
    def __init__(self, z_dim, hidden_node=256, depth=4):
        super(LIIF, self).__init__()
        self.in_f = z_dim * 9 + 4      # feature unfold(*9) / coord concat(+2) / cell size concat(+2)

        self.layers = nn.ModuleList([])
        for i in range(depth):
            dim = self.in_f if i == 0 else (hidden_node + self.in_f)
            self.layers.append(nn.Sequential(
                nn.Linear(dim, hidden_node),
                nn.ReLU()
            ))

    def make_coord(self, shape, ranges=None, flatten=True):
        # Range를 shape등분 한 grid에서 각 cell의 center 좌표 반환
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='xy'), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def forward(self, x, coord, cell):
        # x(feature): (B, z_dim, H, W)
        # coord: (B, sample, 2)
        B, z_dim, H, W = x.shape

        # 주변 3*3 feature concat
        unfold_feat = F.unfold(x, 3, padding=1).view(B, z_dim * 9, H, W)

        vx_lst, vy_lst = [-1, 1], [-1, 1]
        eps_shift = 1e-6

        rh, rw = 1 / H, 1 / W   # Global grid 범위가 (-1, 1)일 때 grid 한 칸의 radius h, w 크기
        feat_coord = self.make_coord(unfold_feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).\
            unsqueeze(0).expand(B, 2, H, W)

        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rh + eps_shift
                coord_[:, :, 1] += vy * rw + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]

                pred = self.layers(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret


class VINR(nn.Module):
    def __init__(self, encoder, liif, modulator, mapper):
        super(VINR, self).__init__()
        self.encoder = encoder
        self.liif = liif
        self.modulator = modulator
        self.mapper = mapper

    def pred_frame(self, frames, t):
        encoded = self.encoder(frames)

        min_v = torch.min(encoded, dim=1, keepdim=True)[0]
        max_v = torch.max(encoded, dim=1, keepdim=True)[0]
        normalized = (((encoded - min_v) / (max_v - min_v + 1e-9)) - 0.5) * 2.0    # (-1, 1)

        b, z_dim, h, w = normalized.shape
        normalized = normalized.permute((0, 2, 3, 1)).view(-1, h*w, z_dim)

        mod_params = self.modulator(normalized)

        rgb = self.mapper(t.unsqueeze(-1), mod_params)
        rgb = rgb.view(-1, h, w, 3).permute(0, 3, 1, 2)

        return rgb

    def pred_frame_from_lr(self, lr_frames, hr_coord, cell, t):
        # LR to HR pred
        lr_encoded = self.encoder(lr_frames)
        hr_encoded_pred = self.liif(lr_encoded, hr_coord, cell)

        min_v = torch.min(hr_encoded_pred, dim=0, keepdim=True)[0]
        max_v = torch.max(hr_encoded_pred, dim=0, keepdim=True)[0]
        hr_encoded_pred_normalized = (((hr_encoded_pred - min_v) / (max_v - min_v)) - 0.5) * 2.0  # (-1, 1)

        hr_mod_params = self.modulator(hr_encoded_pred_normalized)

        hr_rgb = self.mapper(t.unsqueeze(-1), hr_mod_params)

        return hr_rgb

    def forward(self, frames, lr_frames, hr_coord, cell, t):
        hr_rgb_pred = self.pred_frame_from_lr(lr_frames, hr_coord, cell, t)
        pred = self.pred_frame(frames, t)
        return pred, hr_rgb_pred


if __name__ == '__main__':
    batch_size = 7
    num_frames = 5
    z_dim = 50
    h, w = 96, 96
    encoder = XVFIEncoder(in_c=3, num_frames=num_frames, nf=z_dim, n_blocks=2)
    z = encoder(torch.randn((batch_size, 3, num_frames, h, w)))

    coord = None
    cell = None
    liif = LIIF(z_dim)
    continuous_feature = liif(z, coord, cell)

    # modulator = Modulator(continuous_feature, 256, 4)
    # mod_params = modulator(z)
    #
    # mapper = ModRGBMapper(out_dim=3)
    # rgb = mapper(torch.rand(4, 1), mod_params)
    #
    # model = VINR(encoder, modulator, mapper)
    # out = model(torch.rand((4, 3, num_frame, 32, 32)), torch.rand(4))
    # print(out.shape)
