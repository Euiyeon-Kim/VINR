import os
import shutil

import torch.cuda
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from config import Config
from trainer import train
from dataloader import get_dataloader
from common_model import Encoder
from liif_model import VINR


if __name__ == '__main__':
    opt = Config()
    os.makedirs(opt.exp_dir, exist_ok=True)
    shutil.copy('../config.py', f'{opt.exp_dir}/config.py')

    model = None

    train_dataloader, val_dataloader = get_dataloader(opt, opt.mode)

    train(opt, model, train_dataloader, val_dataloader)

