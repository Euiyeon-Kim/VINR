import os
import shutil

from torch import nn

from config import Config
from liif_trainer import train
from dataloader import get_dataloader
from mod_model import VINRDataParallel
from liif_model import XVFIEncoder, LIIF, ModRGBMapper, VINR


if __name__ == '__main__':
    opt = Config()
    opt.model = 'liif'
    os.makedirs(opt.exp_dir, exist_ok=True)
    shutil.copy('config.py', f'{opt.exp_dir}/config.py')

    encoder = XVFIEncoder(in_c=3, num_frames=opt.num_frames, nf=opt.z_dim, n_blocks=2)
    liif = LIIF(opt.z_dim)
    mapper = ModRGBMapper(out_dim=3)
    model = VINR(encoder, liif, mapper)
    # model = nn.DataParallel(model)
    model = VINRDataParallel(model)

    train_dataloader, val_dataloader = get_dataloader(opt)

    train(opt, model, train_dataloader, val_dataloader)

