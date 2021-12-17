import math
import random
from glob import glob


import torch
import numpy as np
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader


def get_dataloader(opt, mode):
    if opt.model == 'liif':
        train_dataset = X4KLIIF(f'{opt.data_root}/train', opt.num_frames, opt.patch_size,
                                scale_max=opt.scale_max, lr_size=opt.lr_size)
        val_dataset = X4KLIIF(f'{opt.data_root}/val', opt.num_frames, opt.patch_size, False,
                              scale_max=opt.scale_max, lr_size=opt.lr_size)
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False,
                                      num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=False,
                                    num_workers=opt.num_workers)

    elif opt.model == 'mod':
        train_dataset = X4K1000FPS(f'{opt.data_root}/train', opt.num_frames, opt.patch_size)
        val_dataset = X4K1000FPS(f'{opt.data_root}/val', opt.num_frames, opt.patch_size, False)
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False,
                                      num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=False,
                                    num_workers=opt.num_workers)

    return train_dataloader, val_dataloader


class X4K1000FPS(Dataset):
    def __init__(self, data_root, num_frames, patch_size, is_train=True):
        super(X4K1000FPS, self).__init__()
        self.is_train = is_train
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.clips = glob(f'{data_root}/*/*')
        self.total_frame = 65 if self.is_train else 33

    def __len__(self):
        return len(self.clips)

    def augment(self, frames, target_frame, target_t):
        # Reverse
        if random.random() < 0.5:
            target_t = 1 - target_t
            frames.reverse()

        frames = np.stack(frames + [target_frame], axis=0)

        # Patchify
        h, w, c = target_frame.shape
        ix = random.randrange(0, w - self.patch_size + 1)
        iy = random.randrange(0, h - self.patch_size + 1)
        frames = frames[:, iy:iy + self.patch_size, ix:ix + self.patch_size, :]

        # Flip
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :]

        # Rotate
        rot = random.randint(0, 3)
        frames = np.rot90(frames, rot, (1, 2))

        return frames, target_t

    def __getitem__(self, item):
        cur_clip = self.clips[item]
        frame_paths = natsorted(glob(f'{cur_clip}/*.png'))
        assert len(frame_paths) == self.total_frame, f'Dataset is not complete. Check {cur_clip}'

        # Set options
        td = random.randint(2, int((self.total_frame - 1) / (self.num_frames - 1)))
        first_frame_idx = random.randint(0, (self.total_frame - 1) - ((self.num_frames - 1) * td))
        last_frame_idx = first_frame_idx + ((self.num_frames - 1) * td)
        selected_idx = np.linspace(first_frame_idx, last_frame_idx, self.num_frames).astype(int)
        target_idx = random.randint(first_frame_idx, last_frame_idx)

        # Read frames
        frames = []
        for idx in selected_idx:
            frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        target_frame = np.array(Image.open(frame_paths[target_idx]).convert('RGB'))
        target_t = (target_idx - first_frame_idx) / (last_frame_idx - first_frame_idx)

        if self.is_train:
            frames, target_t = self.augment(frames, target_frame, target_t)
        else:
            frames = np.stack(frames + [target_frame], axis=0)

        frames = frames.transpose((3, 0, 1, 2)) / 127.5 - 1
        frames = torch.Tensor(frames.astype(float))

        return frames[:, :-1, :, :], frames[:, -1, :, :], target_t


class X4KLIIF(Dataset):
    def __init__(self, data_root, num_frames, patch_size, is_train=True, scale_min=1, scale_max=4, lr_size=96):
        super(X4KLIIF, self).__init__()
        self.is_train = is_train
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.lr_size = lr_size
        self.sample_q = lr_size * lr_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.clips = glob(f'{data_root}/*/*')
        self.total_frame = 65 if self.is_train else 33

    def __len__(self):
        return len(self.clips)

    def augment(self, hr_frames, target_frame, target_t):
        # Reverse
        if random.random() < 0.5:
            target_t = 1 - target_t
            hr_frames.reverse()

        frames = np.stack(hr_frames + [target_frame], axis=0)

        # Patchify
        h, w, c = target_frame.shape
        ix = random.randrange(0, w - self.patch_size + 1)
        iy = random.randrange(0, h - self.patch_size + 1)
        frames = frames[:, iy:iy + self.patch_size, ix:ix + self.patch_size, :]

        # Flip
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :]

        # Rotate
        rot = random.randint(0, 3)
        frames = np.rot90(frames, rot, (1, 2))

        return frames, target_t

    def make_coord(self, shape, ranges=None, flatten=True):
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='xy'), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def to_pixel_samples(self, img):
        coord = self.make_coord(img.shape[-2:])
        rgb = img.view(3, -1).permute(1, 0)
        return coord, rgb

    def make_liif_data(self, frames, scale):
        # frames: T, H, W, C
        lr_res = self.lr_size
        hr_res = round(lr_res * scale)

        # Make HR patch
        _, h, w, _ = frames.shape
        ix = random.randrange(0, w - hr_res + 1)
        iy = random.randrange(0, h - hr_res + 1)
        hr_frames = frames[:, iy:iy + hr_res, ix:ix + hr_res, :]

        inp_frame = hr_frames[:-1, :, :, :]
        target_frame = hr_frames[-1, :, :, :]

        lr_frames = []
        for f in inp_frame:
            lr_f = np.array(Image.fromarray(f).resize((lr_res, lr_res), Image.BICUBIC))
            lr_frames.append(lr_f)
        lr_frames = np.stack(lr_frames, axis=0)

        target_frame = target_frame.transpose((2, 0, 1)) / 127.5 - 1
        target_frame = torch.Tensor(target_frame.astype(float))
        target_coord, target_rgb = self.to_pixel_samples(target_frame.contiguous())

        # Sample target hr frame info to make batch
        sample_lst = np.random.choice(len(target_coord), self.sample_q, replace=False)
        target_coord = target_coord[sample_lst]
        target_rgb = target_rgb[sample_lst]

        cell = torch.ones_like(target_coord)
        cell[:, 0] *= 2 / target_frame.shape[-2]
        cell[:, 1] *= 2 / target_frame.shape[-1]

        return lr_frames, target_coord, target_rgb, cell

    def norm_and_to_tensor(self, img_arr):
        # T, H, W, C -> C, T, H, W
        img_arr = img_arr.transpose((3, 0, 1, 2)) / 127.5 - 1
        img_arr = torch.Tensor(img_arr.astype(float))
        return img_arr

    def __getitem__(self, item):
        cur_clip = self.clips[item]
        frame_paths = natsorted(glob(f'{cur_clip}/*.png'))
        assert len(frame_paths) == self.total_frame, f'Dataset is not complete. Check {cur_clip}'

        # Set options
        td = random.randint(2, int((self.total_frame - 1) / (self.num_frames - 1)))
        first_frame_idx = random.randint(0, (self.total_frame - 1) - ((self.num_frames - 1) * td))
        last_frame_idx = first_frame_idx + ((self.num_frames - 1) * td)
        selected_idx = np.linspace(first_frame_idx, last_frame_idx, self.num_frames).astype(int)
        target_idx = random.randint(first_frame_idx, last_frame_idx)

        # Read frames
        origin_frame = []
        for idx in selected_idx:
            origin_frame.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        origin_target_frame = np.array(Image.open(frame_paths[target_idx]).convert('RGB'))
        target_t = (target_idx - first_frame_idx) / (last_frame_idx - first_frame_idx)

        if self.is_train:
            frames, target_t = self.augment(origin_frame, origin_target_frame, target_t)
        else:
            frames = np.stack(origin_frame + [origin_target_frame], axis=0)

        scale = random.uniform(self.scale_min, self.scale_max)
        lr_frames, target_coord, target_rgb, cell = self.make_liif_data(frames, scale)

        frames = self.norm_and_to_tensor(frames)
        lr_frames = self.norm_and_to_tensor(lr_frames)

        return {
            'input_frames': frames[:, :-1, :, :],
            'target_frame': frames[:, -1, :, :],
            'target_t': target_t,
            'liif_inp': lr_frames,
            'liif_rgb': target_rgb,
            'liif_coord': target_coord,
            'liif_cell': cell,
            'scale': scale
        }


if __name__ == '__main__':
    from config import Config
    config = Config()
    t, v = get_dataloader(config, config.mode)

    for d in t:
        print(d['liif_inp'].shape)
        break
