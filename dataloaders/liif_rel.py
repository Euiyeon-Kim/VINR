import random
from glob import glob
from natsort import natsorted

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from dataloaders import register
from dataloaders.utils import make_liif_data, norm_img_arr_and_to_tensor, augment_frame, \
    augment_t, sample_t, get_rel_ts

"""
    (-1, 1)로 input frame 초기화
    input으로 주는 feature에 상대적으로 target t return
    
    ex) input: [-1, 0, 1]초의 frame target: 0.5초의 frame
        --> target t = 1.5, 0.5, -0.5
"""


class X4KLIIFRel(Dataset):
    def __init__(self, opt, root, is_train=True):
        super(X4KLIIFRel, self).__init__()
        self.opt = opt
        self.num_frames = opt.num_frames
        self.patch_size = opt.patch_size
        self.clips = glob(f'{root}/*/*')
        self.is_train = is_train
        self.selected_ts = torch.Tensor(np.linspace(-1, 1, self.num_frames)).float()
        self.test_inp_idxs = np.linspace(0, 32, self.num_frames).astype(int)
        self.test_ts = torch.Tensor(np.linspace(-1, 1, 33)).float()
        self.test_rel_ts = (self.test_ts - torch.unsqueeze(self.selected_ts, -1).expand(self.num_frames, 33)).permute(1, 0)

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

        # TODO
        if self.opt.test_patch:
            _, h, w, c = target_frames.shape
            ix = random.randrange(0, w - self.opt.test_patch + 1)
            iy = random.randrange(0, h - self.opt.test_patch + 1)
            frames = frames[:, iy:iy + self.opt.test_patch, ix:ix + self.opt.test_patch, :]
            target_frames = target_frames[:, iy:iy + self.opt.test_patch, ix:ix + self.opt.test_patch, :]

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

        return frames, self.selected_ts, self.test_rel_ts, target_coords, target_rgbs, cells, clip_dir

    def __getitem__(self, item):
        cur_clip = self.clips[item]
        frame_paths = natsorted(glob(f'{cur_clip}/*.png'))

        if not self.is_train:       # Validate dataloader
            return self.get_test_item(frame_paths)

        selected_idxs, target_idx = sample_t(self.num_frames, len(frame_paths))
        rel_target_ts, target_t = get_rel_ts(selected_idxs, self.selected_ts, target_idx)

        # Read frames
        origin_frames = []
        for idx in selected_idxs:
            origin_frames.append(np.array(Image.open(frame_paths[idx]).convert('RGB')))
        origin_target_frame = np.array(Image.open(frame_paths[target_idx]).convert('RGB'))

        origin_frames, rel_target_ts = augment_t(origin_frames, rel_target_ts, zero_centered=True, rel_t=True)
        frames = augment_frame(origin_frames, origin_target_frame, self.patch_size)

        target_coord, target_rgb, cell = make_liif_data(frames[-1, :, :, :])

        frames = norm_img_arr_and_to_tensor(frames)
        return {
            'inp_frames': frames[:, :-1, :, :],
            'selected_ts': self.selected_ts,
            'target_t': target_t,
            'rel_target_ts': rel_target_ts,
            'target_coord': target_coord,
            'target_rgb': target_rgb,
            'cell': cell
        }


@register('liif_rel')
def make_mod_dataloader(common_opt, specified_opt):
    train_dataset = X4KLIIFRel(common_opt, f'{common_opt.data_root}/train', True)
    val_dataset = X4KLIIFRel(common_opt, f'{common_opt.data_root}/val', False)
    return train_dataset, val_dataset
