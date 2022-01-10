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
        input_frames, target_frames, target_ts, clip_name = data
        os.makedirs(f'{exp_dir}/val/{epoch}/{clip_name[0]}', exist_ok=True)

        input_frames = input_frames.to(device)
        target_frames = torch.unsqueeze(torch.squeeze(target_frames), 1).to(device)
        target_ts = target_ts.transpose(1, 0).to(device)
        num_t, _ = target_ts.shape

        cur_psnr = 0.
        with torch.no_grad():
            feat = model.get_feat(input_frames)
            for t, f in zip(target_ts, target_frames):
                pred_frame = model.get_rgb(feat, t)
                cur_psnr += psnr(f, pred_frame)
                save_rgbtensor(pred_frame[0], f'{exp_dir}/val/{epoch}/{clip_name[0]}/{t.item():.5f}.png')

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
            input_frames, target_frame, target_t = data['inp_frames'], data['target_frame'], data['target_t']

            pred_frame = model(input_frames, target_t)
            loss = loss_fn(pred_frame, target_frame)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('recon', loss.item(), epoch * steps_per_epoch + step)

            if step % opt.viz_steps == 0:
                print(loss.item())
                save_rgbtensor(target_frame[0], f'{exp_dir}/imgs/{epoch}_{step}_gt_{target_t[0].item():04f}.png')
                save_rgbtensor(pred_frame[0], f'{exp_dir}/imgs/{epoch}_{step}_pred.png')

                viz_input = input_frames[0].permute(1, 0, 2, 3)
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
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(val_psnr)

        if epoch != 0 and epoch % opt.save_epoch == 0:
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict()},
                       f'{exp_dir}/ckpt/{epoch}.pth')


@register('mod')
def make_train_f(common_opt, exp_name):
    train_f = partial(train, common_opt, f'exps/{exp_name}')
    return train_f
