import os
from functools import partial

import numpy as np

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from trainers import register
from trainers.utils import psnr, save_rgbtensor


def warp_loss(warped, target):
    B, C, F, H, W = warped.shape
    gt = torch.unsqueeze(target, 2).repeat(1, 1, F, 1, 1)
    return torch.nn.L1Loss()(gt, warped)


def validate(exp_dir, device, model, val_dataloader, epoch, viz=False):
    model.eval()
    total_psnr = 0.
    for data in val_dataloader:
        input_frames, selected_ts, target_ts, target_coords, target_rgbs, cells, clip_name = data
        if viz:
            os.makedirs(f'{exp_dir}/val/{epoch}/{clip_name[0]}', exist_ok=True)
        input_frames = input_frames.to(device)
        selected_ts = selected_ts.to(device)
        target_ts = torch.unsqueeze(target_ts.transpose(1, 0).float(), -1).to(device)
        target_coords = target_coords.to(device).permute(1, 0, 2, 3)
        target_rgbs = target_rgbs.to(device).permute(1, 0, 2, 3)
        cells = cells.to(device).permute(1, 0, 2, 3)
        num_t, _, _ = target_ts.shape

        cur_psnr = 0.
        with torch.no_grad():
            feat = model.get_feat(input_frames)
            for cnt, (t, rgb, coord, cell) in enumerate(zip(target_ts, target_rgbs, target_coords, cells)):
                pred_frame, _, _ = model.get_rgb(input_frames, feat, coord, cell, selected_ts, t)
                rgb = rgb.contiguous().view(-1, 512, 512, 3).permute(0, 3, 1, 2)
                cur_psnr += psnr(rgb, pred_frame)
                if viz:
                    save_rgbtensor(pred_frame[0], f'{exp_dir}/val/{epoch}/{clip_name[0]}/{cnt}.png')

        total_psnr = total_psnr + (cur_psnr / num_t)

    total_psnr /= len(val_dataloader)
    print(total_psnr.item())
    return total_psnr.item()


def train(opt, exp_dir, model, train_dataloader, val_dataloader):
    os.makedirs(f'{exp_dir}/flow', exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter(f'{exp_dir}/logs')

    model = model.to(device)
    steps_per_epoch = len(train_dataloader)

    best_psnr = 0.
    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.25)

    for epoch in range(opt.epochs):
        model.train()
        cur_epoch_loss = []
        for step, data in enumerate(train_dataloader):
            for k, v in data.items():
                data[k] = v.to(device)

            inp_frames, selected_ts, target_t, target_coord, target_rgb, cell = \
                data['inp_frames'], data['selected_ts'], data['target_t'], data['target_coord'], data['target_rgb'], data['cell']

            pred_frame, warped, visibilities = model(inp_frames, target_coord, cell, selected_ts, target_t)
            target_frame = data['target_rgb'].contiguous().view(-1, opt.patch_size, opt.patch_size, 3).permute(0, 3, 1, 2)

            loss = loss_fn(pred_frame, target_frame)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_epoch_loss.append(loss.item())

            if step % opt.viz_steps == 0:
                save_rgbtensor(pred_frame[0], f'{exp_dir}/imgs/{epoch}_{step}_pred.png')
                save_rgbtensor(target_frame[0], f'{exp_dir}/imgs/{epoch}_{step}_gt_{target_t[0].item():04f}.png')

                viz_input = inp_frames[0].permute(1, 0, 2, 3)
                viz_input = (viz_input + 1.) / 2.
                for idx, img in enumerate(viz_input):
                    save_rgbtensor(img, f'{exp_dir}/imgs/{epoch}_{step}_{idx}.png', norm=False)

                viz_warp = warped[0].permute(1, 0, 2, 3)
                viz_warp = (viz_warp + 1.) / 2.
                for idx, img in enumerate(viz_warp):
                    save_rgbtensor(img, f'{exp_dir}/flow/{epoch}_{step}_w{idx}.png', norm=False)

                viz_mask = visibilities[0].permute(2, 0, 1)
                for idx, img in enumerate(viz_mask):
                    save_rgbtensor(torch.unsqueeze(img, 0),
                                   f'{exp_dir}/flow/{epoch}_{step}_m{idx}_{torch.mean(img):04f}.png', norm=False)

        print(np.mean(cur_epoch_loss))
        writer.add_scalar('train/loss', np.mean(cur_epoch_loss), epoch)

        # Validate - save best model
        viz = True if (epoch + 1) % opt.val_save_epoch == 0 else False
        val_psnr = validate(exp_dir, device, model, val_dataloader, epoch, viz)

        writer.add_scalar('val_psnr', val_psnr, epoch)
        if val_psnr > best_psnr:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{exp_dir}/ckpt/best.pth')
            best_psnr = val_psnr

        # Log lr
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()

        # Save model periodically
        if (epoch + 1) % opt.save_epoch == 0:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{exp_dir}/ckpt/{epoch+1}.pth')


@register('liif_flow_v2')
def make_train_f(common_opt, exp_name):
    train_f = partial(train, common_opt, f'exps/{exp_name}')
    return train_f
