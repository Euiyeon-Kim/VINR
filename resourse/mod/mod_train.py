import os
import shutil

from config import Config
from mod_trainer import train
from dataloader import get_dataloader
from models.common import Encoder
from models.mod import Modulator, ModRGBMapper, VINR, VINRDataParallel


if __name__ == '__main__':
    opt = Config()
    opt.model = 'mod'
    os.makedirs(opt.exp_dir, exist_ok=True)
    shutil.copy('../../config.py', f'{opt.exp_dir}/config.py')

    encoder = Encoder(in_dim=3*opt.num_frames, out_dim=opt.z_dim)
    modulator = Modulator(in_f=opt.z_dim, hidden_node=256, depth=4)
    mapper = ModRGBMapper(out_dim=3, hidden_node=256, depth=5)

    model = VINR(encoder, modulator, mapper)
    # model = nn.DataParallel(model)
    model = VINRDataParallel(model)

    train_dataloader, val_dataloader = get_dataloader(opt)

    train(opt, model, train_dataloader, val_dataloader)
