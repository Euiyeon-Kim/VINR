import os

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from trainer import save_rgbtensor, psnr


def validate(opt, device, model, val_dataloader, epoch):
    model.eval()
    total_psnr = 0.
    for data in val_dataloader:
        input_frames, target_frames, target_ts, coord, rgb, cell, clip_name = data
        os.makedirs(f'{opt.exp_dir}/val/{epoch}/{clip_name[0]}', exist_ok=True)

        input_frames = input_frames.to(device)
        target_frames = torch.unsqueeze(torch.squeeze(target_frames), 1).to(device)
        target_ts = target_ts.transpose(1, 0).float().to(device)
        num_t, _ = target_ts.shape

        cur_psnr = 0.
        with torch.no_grad():
            feat = model.get_feat(input_frames)
            for t, f in zip(target_ts, target_frames):
                pred_frame = model.get_rgb(feat, t)
                cur_psnr += psnr(f, pred_frame)
                save_rgbtensor(pred_frame[0], f'{opt.exp_dir}/val/{epoch}/{clip_name[0]}/{t.item():.5f}.png')

        total_psnr = total_psnr + (cur_psnr / num_t)

    total_psnr /= len(val_dataloader)
    print(total_psnr.item())
    return total_psnr.item()


def train(opt, model, train_dataloader, val_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(f'{opt.exp_dir}/logs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/imgs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/ckpt', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/val', exist_ok=True)
    writer = SummaryWriter(f'{opt.exp_dir}/logs')

    model = model.to(device)

    steps_per_epoch = len(train_dataloader)

    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=opt.min_lr)

    for epoch in range(opt.epochs):
        model.train()
        for step, data in enumerate(train_dataloader):
            inp_frames, target_frame, t, coord, rgb, cell = data
            inp_frames = inp_frames.to(device)
            target_frame = target_frame.to(device)
            t = torch.unsqueeze(t.float(), -1).to(device)
            coord = coord.to(device)
            rgb = rgb.to(device)
            cell = cell.to(device)

            pred_frame = model(inp_frames, coord, cell, t)
            loss = loss_fn(pred_frame, rgb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('recon', loss.item(), epoch * steps_per_epoch + step)
            if step % opt.viz_step == 0:
                print(loss.item())
                save_rgbtensor(target_frame[0], f'{opt.exp_dir}/imgs/{epoch}_{step}_gt_{t[0]:04f}.png')

        # Log lr
        writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], epoch)
        scheduler.step()

        # Save best psnr model
        if (epoch + 1) % opt.save_epoch == 0:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{opt.exp_dir}/ckpt/{epoch+1}.pth')




