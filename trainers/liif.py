import os
from functools import partial

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from trainers import register
from trainers.utils import psnr, save_rgbtensor


def validate(exp_dir, device, model, val_dataloader, epoch):
    model.eval()
    total_psnr = 0.
    for data in val_dataloader:
        input_frames, target_ts, target_coords, target_rgbs, cells, clip_name = data
        os.makedirs(f'{exp_dir}/val/{epoch}/{clip_name[0]}', exist_ok=True)
        input_frames = input_frames.to(device)
        target_ts = torch.unsqueeze(target_ts.transpose(1, 0).float(), -1).to(device)
        target_coords = target_coords.to(device).permute(1, 0, 2, 3)
        target_rgbs = target_rgbs.to(device).permute(1, 0, 2, 3)
        cells = cells.to(device).permute(1, 0, 2, 3)
        num_t, _, _ = target_ts.shape

        cur_psnr = 0.
        with torch.no_grad():
            feat = model.get_feat(input_frames)
            for t, rgb, coord, cell in zip(target_ts, target_rgbs, target_coords, cells):
                pred_frame = model.get_rgb(feat, coord, cell, t)
                pred_frame = pred_frame.contiguous().view(512, 512, 3)
                rgb = rgb.contiguous().view(512, 512, 3)
                cur_psnr += psnr(rgb, pred_frame)
                viz_pred = pred_frame.permute(2, 0, 1)
                save_rgbtensor(viz_pred, f'{exp_dir}/val/{epoch}/{clip_name[0]}/{t.item():.5f}.png')

        total_psnr = total_psnr + (cur_psnr / num_t)

    total_psnr /= len(val_dataloader)
    print(total_psnr.item())
    return total_psnr.item()


def train(opt, exp_dir, model, train_dataloader, val_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter(f'{exp_dir}/logs')

    model = model.to(device)
    steps_per_epoch = len(train_dataloader)

    best_psnr = 0.
    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=opt.factor,
                                               patience=opt.patience, min_lr=opt.min_lr)

    for epoch in range(opt.epochs):
        model.train()
        for step, data in enumerate(train_dataloader):
            for k, v in data.items():
                data[k] = v.to(device)

            inp_frames, target_t, target_coord, target_rgb, cell = \
                data['inp_frames'], data['target_t'], data['target_coord'], data['target_rgb'], data['cell']

            pred_frame = model(inp_frames, target_coord, cell, target_t)

            loss = loss_fn(pred_frame, target_rgb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('recon', loss.item(), epoch * steps_per_epoch + step)
            if step % opt.viz_steps == 0:
                print(loss.item())
                viz_pred = pred_frame[0].contiguous().view(opt.patch_size, opt.patch_size, 3).permute(2, 0, 1)
                viz_target = target_rgb[0].contiguous().view(opt.patch_size, opt.patch_size, 3).permute(2, 0, 1)
                save_rgbtensor(viz_pred, f'{exp_dir}/imgs/{epoch}_{step}_pred.png')
                save_rgbtensor(viz_target, f'{exp_dir}/imgs/{epoch}_{step}_gt_{target_t[0].item():04f}.png')

                viz_input = inp_frames[0].permute(1, 0, 2, 3)
                viz_input = (viz_input + 1.) / 2.
                for idx, img in enumerate(viz_input):
                    save_rgbtensor(img, f'{exp_dir}/imgs/{epoch}_{step}_{idx}.png', norm=False)

        # Validate - save best model
        val_psnr = validate(exp_dir, device, model, val_dataloader, epoch)
        writer.add_scalar('val_psnr', val_psnr, epoch)
        if val_psnr > best_psnr:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{exp_dir}/ckpt/best.pth')
            best_psnr = val_psnr

        # Log lr
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(val_psnr)

        # Save best psnr model
        if (epoch + 1) % opt.save_epoch == 0:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{exp_dir}/ckpt/{epoch+1}.pth')


@register('liif')
def make_train_f(common_opt, exp_name):
    train_f = partial(train, common_opt, f'exps/{exp_name}')
    return train_f
