import argparse

import torch
from torch import nn

from common_model import Encoder
from mod_model import Modulator, ModRGBMapper, VINR
from dataloader import X4K1000FPS

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_name', default='frame5_mod_z128')
args = parser.parse_args()
EXP_NAME = args.exp_name


if __name__ == '__main__':
    from test_config import Config
    opt = Config()

    encoder = Encoder(in_dim=3*opt.num_frames, out_dim=opt.z_dim)
    modulator = Modulator(in_f=opt.z_dim, hidden_node=256, depth=4)
    mapper = ModRGBMapper(out_dim=3, hidden_node=256, depth=5)
    model = VINR(encoder, modulator, mapper)
    model = nn.DataParallel(model)

    ckpt = torch.load(f'exps/{EXP_NAME}/ckpt/best.pth')
    model.load_state_dict(ckpt['model'])

    test_dataset = X4K1000FPS(opt, False)

