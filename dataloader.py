import math
import random
from glob import glob

import torch
import numpy as np
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader


def get_dataloader(opt):
    dataset = X4KLIIF if opt.model == 'liif' else X4K1000FPS
    train_dataset = dataset(opt, f'{opt.data_root}/train', True)
    val_dataset = dataset(opt, f'{opt.data_root}/val', False)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                      drop_last=False, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=1)
    return train_dataloader, val_dataloader


class X4K1000FPS(Dataset):
    def __init__(self, opt, root, is_train=True):
        super(X4K1000FPS, self).__init__()
        self.is_train = is_train
        self.num_frames = opt.num_frames
        self.patch_size = opt.patch_size
        self.clips = glob(f'{root}/*/*')
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

    def get_test_item(self, frame_paths):
        clip_dir = frame_paths[0].split('/')[-2]
        selected_idx = np.linspace(0, 32, self.num_frames).astype(int)
        target_idxs = np.arange(33)

        frames = []
        for idx in selected_idx:
            frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        for idx in target_idxs:
            frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        frames = np.stack(frames, axis=0)

        target_ts = target_idxs / 32.

        frames = frames.transpose((0, 3, 1, 2)) / 127.5 - 1
        frames = torch.Tensor(frames.astype(float))

        inp_frames = frames[:self.num_frames, :, :, :].permute((1, 0, 2, 3))
        target_frames = frames[self.num_frames:, :, :, :]

        return inp_frames, target_frames, target_ts, clip_dir

    def __getitem__(self, item):        # B, C, T, H, W
        cur_clip = self.clips[item]
        frame_paths = natsorted(glob(f'{cur_clip}/*.png'))
        assert len(frame_paths) == self.total_frame, f'Dataset is not complete. Check {cur_clip}'

        if not self.is_train:       # Test and validate dataloader
            return self.get_test_item(frame_paths)

        # Set train options
        max_td = 32 if self.num_frames == 2 else int((self.total_frame - 1) / (self.num_frames - 1))
        td = random.randint(2, max_td)
        first_frame_idx = random.randint(0, (self.total_frame - 1) - ((self.num_frames - 1) * td))
        last_frame_idx = first_frame_idx + ((self.num_frames - 1) * td)
        selected_idx = np.linspace(first_frame_idx, last_frame_idx, self.num_frames).astype(int)
        target_idx = random.randint(first_frame_idx, last_frame_idx)
        # scale = random.uniform(1, 2)

        # Read frames
        frames = []
        for idx in selected_idx:
            # f = Image.open(frame_paths[idx]).convert('RGB')
            # h, w = f.size
            # small_h, small_w = int(h / scale), int(w / scale)
            # frames.append(np.array(f.resize((small_h, small_w))))
            frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))

        # target_frame = np.array(Image.open(frame_paths[target_idx]).convert('RGB').resize((small_h, small_w)))
        target_frame = np.array(Image.open(frame_paths[target_idx]).convert('RGB'))
        target_t = (target_idx - first_frame_idx) / (last_frame_idx - first_frame_idx)

        frames, target_t = self.augment(frames, target_frame, target_t)

        frames = frames.transpose((3, 0, 1, 2)) / 127.5 - 1
        frames = torch.Tensor(frames.astype(float))

        return frames[:, :-1, :, :], frames[:, -1, :, :], target_t


class X4KLIIF(Dataset):
    def __init__(self, opt, root, is_train=True):
        super(X4KLIIF, self).__init__()
        self.opt = opt
        self.sample_q = opt.sample_q
        self.num_frames = opt.num_frames
        self.patch_size = opt.patch_size
        self.clips = glob(f'{root}/*/*')
        self.is_train = is_train
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
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def to_pixel_samples(self, img):
        coord = self.make_coord(img.shape[-2:])
        rgb = img.view(3, -1).permute(1, 0)
        return coord, rgb

    def make_liif_data(self, frames):
        # frames: T, H, W, C
        target_frame = frames[-1, :, :, :]

        target_frame = target_frame.transpose((2, 0, 1)) / 127.5 - 1
        target_frame = torch.Tensor(target_frame.astype(float))
        target_coord, target_rgb = self.to_pixel_samples(target_frame.contiguous())

        # Sample target hr frame info to make batch
        if self.sample_q:
            sample_lst = np.random.choice(len(target_coord), self.sample_q, replace=False)
            target_coord = target_coord[sample_lst]
            target_rgb = target_rgb[sample_lst]

        cell = torch.ones_like(target_coord)
        cell[:, 0] *= 2 / target_frame.shape[-2]
        cell[:, 1] *= 2 / target_frame.shape[-1]
        return target_coord, target_rgb, cell

    def norm_and_to_tensor(self, img_arr):
        # T, H, W, C -> C, T, H, W
        img_arr = img_arr.transpose((3, 0, 1, 2)) / 127.5 - 1
        img_arr = torch.Tensor(img_arr.astype(float))
        return img_arr

    def get_test_item(self, frame_paths):
        clip_dir = frame_paths[0].split('/')[-2]
        selected_idx = np.linspace(0, 32, self.num_frames).astype(int)
        target_idxs = np.arange(33)

        frames = []
        for idx in selected_idx:
            frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        for idx in target_idxs:
            frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        frames = np.stack(frames, axis=0)

        target_ts = target_idxs / 32.
        target_coord, target_rgb, cell = self.make_liif_data(frames)

        frames = frames.transpose((0, 3, 1, 2)) / 127.5 - 1
        frames = torch.Tensor(frames.astype(float))

        inp_frames = frames[:self.num_frames, :, :, :].permute((1, 0, 2, 3))
        target_frames = frames[self.num_frames:, :, :, :]

        return inp_frames, target_frames, target_ts, target_coord, target_rgb, cell, clip_dir

    def __getitem__(self, item):
        cur_clip = self.clips[item]
        frame_paths = natsorted(glob(f'{cur_clip}/*.png'))
        assert len(frame_paths) == self.total_frame, f'Dataset is not complete. Check {cur_clip}'

        if not self.is_train:       # Test and validate dataloader
            return self.get_test_item(frame_paths)

        # Set options
        max_td = 32 if self.num_frames == 2 else int((self.total_frame - 1) / (self.num_frames - 1))
        td = random.randint(2, max_td)
        first_frame_idx = random.randint(0, (self.total_frame - 1) - ((self.num_frames - 1) * td))
        last_frame_idx = first_frame_idx + ((self.num_frames - 1) * td)
        selected_idx = np.linspace(first_frame_idx, last_frame_idx, self.num_frames).astype(int)
        target_idx = random.randint(first_frame_idx, last_frame_idx)

        # Read frames
        origin_frames = []
        for idx in selected_idx:
            origin_frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        origin_target_frame = np.array(Image.open(frame_paths[target_idx]).convert('RGB'))
        target_t = (target_idx - first_frame_idx) / (last_frame_idx - first_frame_idx)

        frames, target_t = self.augment(origin_frames, origin_target_frame, target_t)
        target_coord, target_rgb, cell = self.make_liif_data(frames)

        frames = self.norm_and_to_tensor(frames)
        return frames[:, :-1, :, :], frames[:, -1, :, :], target_t, target_coord, target_rgb, cell


if __name__ == '__main__':
    from config import Config
    config = Config()

    config.model = 'mod'
    train, val = get_dataloader(config)
    for d in train:
        inp, target, t = d
        print(inp.shape, target.shape, t.shape)
        break
    for d in val:
        inp, target, t, name = d
        print(inp.shape, target.shape, t.shape, name[0])
        exit()

    config.model = 'liif'   # 'mod'
    train, val = get_dataloader(config)
    for d in train:
        inp, target, t, coord, rgb, cell = d
        print(inp.shape, target.shape, t.shape, coord.shape, rgb.shape, cell.shape)
        break
    for d in val:
        inp, target, t, coord, rgb, cell, name = d
        print(inp.shape, target.shape, t.shape, coord.shape, rgb.shape, cell.shape, name[0])
        exit()
