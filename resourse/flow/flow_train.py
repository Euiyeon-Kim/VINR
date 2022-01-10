import os
import shutil

from torch import nn

from config import Config
from flow_trainer import train
from dataloader import get_dataloader
from models.common import Encoder
from models.mod import Modulator
from models.flow import ModMapper, VINR


if __name__ == '__main__':
    opt = Config()
    opt.model = 'mod'
    os.makedirs(opt.exp_dir, exist_ok=True)
    shutil.copy('../../config.py', f'{opt.exp_dir}/config.py')

    encoder = Encoder(in_dim=3*opt.num_frames, out_dim=opt.z_dim)
    modulator = Modulator(in_f=opt.z_dim, hidden_node=256, depth=4)
    mod_generator = ModMapper(out_dim=opt.num_frames*3, hidden_node=256, depth=5)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VINR(encoder, modulator, mod_generator, device, opt.num_frames)
    model = nn.DataParallel(model)

    train_dataloader, val_dataloader = get_dataloader(opt)

    train(opt, model, train_dataloader, val_dataloader)
