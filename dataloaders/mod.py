import random
from glob import glob
from natsort import natsorted

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from dataloaders import register


class X4K1000FPS(Dataset):
    """
        inp_frames: B, C, num_frames, H, W
        target_frame: B, C, H, W
        target_t: B
    """
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
        target_t = torch.Tensor([target_t]).float()

        return {
            'inp_frames': frames[:, :-1, :, :],
            'target_frame': frames[:, -1, :, :],
            'target_t': target_t
        }


@register('mod')
def make_mod_dataloader(common_opt):
    train_dataset = X4K1000FPS(common_opt, f'{common_opt.data_root}/train', True)
    val_dataset = X4K1000FPS(common_opt, f'{common_opt.data_root}/val', False)
    return train_dataset, val_dataset
