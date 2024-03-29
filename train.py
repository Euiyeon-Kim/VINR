import os
import shutil

from config import config
from build_pipeline import prepare_train

if __name__ == '__main__':
    opt = config
    os.makedirs(f'exps/{opt.exp_name}/logs', exist_ok=True)
    os.makedirs(f'exps/{opt.exp_name}/imgs', exist_ok=True)
    os.makedirs(f'exps/{opt.exp_name}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{opt.exp_name}/val', exist_ok=True)
    shutil.copy('config.py', f'exps/{opt.exp_name}/config.py')

    model, train_dataloader, val_dataloader, train_f = prepare_train(opt)

    train_f(model, train_dataloader, val_dataloader)
