import os

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from trainer import save_rgbtensor


def train(opt, model, train_dataloader, val_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(f'{opt.exp_dir}/logs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/imgs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/liif_imgs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/ckpt', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/val', exist_ok=True)
    writer = SummaryWriter(f'{opt.exp_dir}/logs')

    model = model.to(device)

    steps_per_epoch = len(train_dataloader)
    best_psnr = 0

    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=opt.min_lr)

    for epoch in range(opt.epochs):
        model.train()
        for step, data in enumerate(train_dataloader):
            for k, v in data.items():
                data[k] = v.to(device)

            pred_frame, hr_pred = model(data['input_frames'], data['liif_inp'], data['liif_coord'],
                                        data['liif_cell'], data['target_t'].float())

            liif_loss = loss_fn(hr_pred, data['liif_rgb'])
            origin_loss = loss_fn(pred_frame, data['target_frame'])
            loss = opt.liif_lambda * liif_loss + origin_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('total', loss.item(), epoch * steps_per_epoch + step)
            writer.add_scalar('train/liif', liif_loss.item(), epoch * steps_per_epoch + step)
            writer.add_scalar('train/origin', origin_loss.item(), epoch * steps_per_epoch + step)

            if step % opt.viz_step == 0:
                save_rgbtensor(data['target_frame'][0], f'{opt.exp_dir}/imgs/{epoch}_{step}_gt_{data["target_t"][0]:04f}.png')
                save_rgbtensor(pred_frame[0], f'{opt.exp_dir}/imgs/{epoch}_{step}_pred.png')

                viz_input = data['input_frames'][0].permute(1, 0, 2, 3)
                viz_input = (viz_input + 1.) / 2.
                for idx, img in enumerate(viz_input):
                    save_rgbtensor(img, f'{opt.exp_dir}/imgs/{epoch}_{step}_{idx}.png', norm=False)

                viz_liif_target = data['liif_rgb'][0].view(opt.lr_size, opt.lr_size, -1).permute(2, 0, 1)
                viz_liif_pred = hr_pred[0].view(opt.lr_size, opt.lr_size, -1).permute(2, 0, 1)
                save_rgbtensor(viz_liif_target, f'{opt.exp_dir}/liif_imgs/{epoch}_{step}_gt.png')
                save_rgbtensor(viz_liif_pred, f'{opt.exp_dir}/liif_imgs/{epoch}_{step}_pred.png')

        # Log lr
        writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], epoch)
        scheduler.step()

        # Save best psnr model
        if (epoch + 1) % opt.save_epoch == 0:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{opt.exp_dir}/ckpt/{epoch+1}.pth')




