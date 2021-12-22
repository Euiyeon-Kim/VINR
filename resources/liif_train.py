import os
import shutil

from torch import nn

from config import Config
from liif_trainer import train
from dataloader import get_dataloader
from common_model import Encoder
from liif_model import LIIF, VINR, Modulator, ModRGBMapper


if __name__ == '__main__':
    opt = Config()
    opt.model = 'liif'
    os.makedirs(opt.exp_dir, exist_ok=True)
    shutil.copy('../config.py', f'{opt.exp_dir}/config.py')

    encoder = Encoder(in_dim=3 * opt.num_frames, out_dim=opt.z_dim)
    liif = LIIF(opt.z_dim)
    modulator = Modulator(in_f=opt.z_dim, hidden_node=256, depth=4)
    mapper = ModRGBMapper(out_dim=3, hidden_node=256, depth=5)

    model = VINR(encoder, liif, modulator, mapper)
    model = nn.DataParallel(model)

    train_dataloader, val_dataloader = get_dataloader(opt, opt.mode)

    train(opt, model, train_dataloader, val_dataloader)

