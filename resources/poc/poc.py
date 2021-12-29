import os
import shutil

import torch.cuda
from torch import nn
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataloader import get_dataloader
from common_model import Encoder
from poc_model import Reflector, VINR


# def save_img(bgr_tensor, path, norm=True):
#     if norm:
#         bgr_tensor = (bgr_tensor + 1.) / 2.
#     b, g, r = torch.chunk(bgr_tensor, 3, dim=0)
#     rgb = torch.concat((r, g, b), dim=0)
#     save_image(rgb, path)
#
#
# def train(opt):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     os.makedirs(f'{opt.exp_dir}/logs', exist_ok=True)
#     os.makedirs(f'{opt.exp_dir}/imgs', exist_ok=True)
#     os.makedirs(f'{opt.exp_dir}/ckpt', exist_ok=True)
#     writer = SummaryWriter(f'{opt.exp_dir}/logs')
#
#     encoder = Encoder(in_dim=3 * opt.num_frames, out_dim=opt.z_dim)
#     mapper = Reflector(in_dim=1, out_dim=3, z_dim=opt.z_dim)
#
#     model = VINR(encoder, mapper)
#     model = nn.DataParallel(model).to(device)
#
#     train_dataloader, valid_dataloader = get_dataloader(opt)
#
#     train(opt, model, train_dataloader, val_dataloader)

    # loss_fn = nn.L1Loss()
    # optimizer = Adam(model.parameters(), lr=opt.lr)
    # for epoch in range(opt.epochs):
    #     for step, data in enumerate(train_dataloader):
    #         input_frames, target_frame, target_t = data
    #         input_frames = input_frames.to(device)
    #         target_frame = target_frame.to(device)
    #         target_t = target_t.float().to(device)
    #
    #         pred_frame = model(input_frames, target_t)
    #         loss = loss_fn(pred_frame, target_frame)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print(loss.item())
    #         writer.add_scalar('recon', loss.item(), epoch * steps_per_epoch + step)
    #
    #         if step % opt.viz_step == 0:
    #             save_img(target_frame[0], f'{opt.exp_dir}/imgs/{epoch}_{step}_target_{target_t[0]:04f}.png')
    #             save_img(pred_frame[0], f'{opt.exp_dir}/imgs/{epoch}_{step}_pred.png')
    #
    #             viz_input = input_frames[0].permute(1, 0, 2, 3)
    #             viz_input = (viz_input + 1.) / 2.
    #             for idx, img in enumerate(viz_input):
    #                 save_img(img, f'{opt.exp_dir}/imgs/{epoch}_{step}_{idx}.png', norm=False)
    #
    #     if (epoch + 1) % opt.save_epoch == 0:
    #         torch.save(model.state_dict(), f'{opt.exp_dir}/ckpt/{epoch + 1}.pth')


if __name__ == '__main__':
    opt = Config()
    opt.model = 'mod'
    os.makedirs(opt.exp_dir, exist_ok=True)
    shutil.copy('../../config.py', f'{opt.exp_dir}/config.py')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(f'{opt.exp_dir}/logs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/imgs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/ckpt', exist_ok=True)
    writer = SummaryWriter(f'{opt.exp_dir}/logs')

    encoder = Encoder(in_dim=3 * opt.num_frames, out_dim=opt.z_dim)
    mapper = Reflector(in_dim=1, out_dim=3, z_dim=opt.z_dim)

    model = VINR(encoder, mapper)
    model = nn.DataParallel(model).to(device)

    train_dataloader, val_dataloader = get_dataloader(opt)

    from trainer import train
    train(opt, model, train_dataloader, val_dataloader)

    # train(config)
