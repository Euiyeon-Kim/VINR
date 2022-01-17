import random
from glob import glob
from natsort import natsorted

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from dataloaders import register
from dataloaders.utils import make_liif_data, norm_img_arr_and_to_tensor, augment, sample_t, normalize_ts


class X4KLIIFZC(Dataset):
    def __init__(self, opt, root, is_train=True):
        super(X4KLIIFZC, self).__init__()
        self.opt = opt
        self.num_frames = opt.num_frames
        self.patch_size = opt.patch_size
        self.clips = glob(f'{root}/*/*')
        self.is_train = is_train
        self.total_frame = 65 if self.is_train else 33
        self.selected_ts = torch.Tensor(np.linspace(-1, 1, self.num_frames)).float()
        self.test_inp_idxs = np.linspace(0, 32, self.num_frames).astype(int)
        self.test_ts = torch.Tensor(np.linspace(-1, 1, 33)).float()

    def __len__(self):
        return len(self.clips)

    def get_test_item(self, frame_paths):
        clip_dir = frame_paths[0].split('/')[-2]

        frames = []
        for idx in self.test_inp_idxs:
            frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        frames = np.stack(frames, axis=0)

        target_frames = []
        for idx in range(33):
            target_frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        target_frames = np.stack(target_frames, axis=0)

        # # TODO
        # _, h, w, c = target_frames.shape
        # ix = random.randrange(0, w - self.patch_size + 1)
        # iy = random.randrange(0, h - self.patch_size + 1)
        # frames = frames[:, iy:iy + self.patch_size, ix:ix + self.patch_size, :]
        # target_frames = target_frames[:, iy:iy + self.patch_size, ix:ix + self.patch_size, :]

        target_coords, target_rgbs, cells = [], [], []
        for target_frame in target_frames:
            target_coord, target_rgb, cell = make_liif_data(target_frame)
            target_coords.append(target_coord)
            target_rgbs.append(target_rgb)
            cells.append(cell)

        target_coords = torch.stack(target_coords)
        target_rgbs = torch.stack(target_rgbs)
        cells = torch.stack(cells)

        frames = frames.transpose((0, 3, 1, 2)) / 127.5 - 1
        frames = torch.Tensor(frames.astype(float)).permute(1, 0, 2, 3)

        return frames, self.selected_ts, self.test_ts, target_coords, target_rgbs, cells, clip_dir

    def __getitem__(self, item):
        cur_clip = self.clips[item]
        frame_paths = natsorted(glob(f'{cur_clip}/*.png'))
        assert len(frame_paths) == self.total_frame, f'Dataset is not complete. Check {cur_clip}'

        if not self.is_train:       # Test and validate dataloader
            return self.get_test_item(frame_paths)

        selected_idxs, target_idx = sample_t(self.num_frames)
        target_t = normalize_ts(selected_idxs, target_idx, True)

        # Read frames
        origin_frames = []
        for idx in selected_idxs:
            origin_frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        origin_target_frame = np.array(Image.open(frame_paths[target_idx]).convert('RGB'))

        frames, target_t = augment(origin_frames, origin_target_frame, target_t, self.patch_size, True)
        target_coord, target_rgb, cell = make_liif_data(frames[-1, :, :, :])
        target_t = torch.Tensor([target_t]).float()

        frames = norm_img_arr_and_to_tensor(frames)
        return {
            'inp_frames': frames[:, :-1, :, :],
            'selected_ts': self.selected_ts,
            'target_t': target_t,
            'target_coord': target_coord,
            'target_rgb': target_rgb,
            'cell': cell
        }


@register('liif_zero_centered')
def make_mod_dataloader(common_opt, specified_opt):
    train_dataset = X4KLIIFZC(common_opt, f'{common_opt.data_root}/train', True)
    val_dataset = X4KLIIFZC(common_opt, f'{common_opt.data_root}/val', False)
    return train_dataset, val_dataset
