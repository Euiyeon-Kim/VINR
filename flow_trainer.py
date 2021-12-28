import os

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


def save_rgbtensor(rgbtensor, path, norm=True):
    if norm:
        rgbtensor = (rgbtensor + 1.) / 2.
    save_image(rgbtensor, path)


def psnr(gt, pred, norm=True):
    if norm:    # (-1, 1) to (0, 255)
        gt = ((gt + 1.) / 2.) * 255.
        pred = ((pred + 1.) / 2.) * 255.
    diff = gt - pred
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    if rmse == 0:
        return float('inf')
    return 20 * torch.log10(255. / rmse)


def validate(opt, device, model, val_dataloader, epoch):
    cur_psnr = 0
    model.eval()
    for data in val_dataloader:
        input_frames, target_frame, target_t = data
        input_frames = input_frames.to(device)
        target_frame = target_frame.to(device)
        target_t = target_t.float().to(device)

        with torch.no_grad():
            pred_frame, _, _ = model(input_frames, target_t)
            cur_psnr += psnr(target_frame, pred_frame)

    cur_psnr /= len(val_dataloader)
    save_rgbtensor(target_frame[0], f'{opt.exp_dir}/val/{epoch}_gt_{target_t[0]:04f}.png')
    save_rgbtensor(pred_frame[0], f'{opt.exp_dir}/val/{epoch}_pred.png')
    viz_input = input_frames[0].permute(1, 0, 2, 3)
    viz_input = (viz_input + 1.) / 2.
    for idx, img in enumerate(viz_input):
        save_rgbtensor(img, f'{opt.exp_dir}/val/{epoch}_{idx}.png', norm=False)

    return cur_psnr.item()


def train(opt, model, train_dataloader, val_dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(f'{opt.exp_dir}/logs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/imgs', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/ckpt', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/val', exist_ok=True)
    os.makedirs(f'{opt.exp_dir}/flow', exist_ok=True)

    writer = SummaryWriter(f'{opt.exp_dir}/logs')

    model = model.to(device)
    steps_per_epoch = len(train_dataloader)

    best_psnr = 0

    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=opt.min_lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=5, min_lr=opt.min_lr)

    for epoch in range(opt.epochs):
        model.train()
        for step, data in enumerate(train_dataloader):
            input_frames, target_frame, target_t = data

            input_frames = input_frames.to(device)
            target_frame = target_frame.to(device)
            target_t = target_t.float().to(device)

            pred_frame, warped, visibilities = model(input_frames, target_t)
            loss = loss_fn(pred_frame, target_frame)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('recon', loss.item(), epoch * steps_per_epoch + step)

            if step % opt.viz_step == 0:
                print(loss.item())
                save_rgbtensor(target_frame[0], f'{opt.exp_dir}/imgs/{epoch}_{step}_gt_{target_t[0]:04f}.png')
                save_rgbtensor(pred_frame[0], f'{opt.exp_dir}/imgs/{epoch}_{step}_pred.png')

                viz_input = input_frames[0].permute(1, 0, 2, 3)
                viz_input = (viz_input + 1.) / 2.
                for idx, img in enumerate(viz_input):
                    save_rgbtensor(img, f'{opt.exp_dir}/imgs/{epoch}_{step}_{idx}.png', norm=False)

                viz_warp = warped[0].permute(1, 0, 2, 3)
                viz_warp = (viz_warp + 1.) / 2.
                for idx, img in enumerate(viz_warp):
                    save_rgbtensor(img, f'{opt.exp_dir}/flow/{epoch}_{step}_w{idx}.png', norm=False)

                viz_mask = visibilities[0].permute(2, 0, 1)
                for idx, img in enumerate(viz_mask):
                    save_image(torch.unsqueeze(img, 0),
                               f'{opt.exp_dir}/flow/{epoch}_{step}_m{idx}_{torch.mean(img):04f}.png')

        # Validate - save best model
        val_psnr = validate(opt, device, model, val_dataloader, epoch)
        writer.add_scalar('val_psnr', val_psnr, epoch)
        if val_psnr > best_psnr:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{opt.exp_dir}/ckpt/best.pth')
            best_psnr = val_psnr

        # Log lr
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(val_psnr)

        if (epoch+1) % opt.save_epoch == 0:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{opt.exp_dir}/ckpt/{epoch}.pth')