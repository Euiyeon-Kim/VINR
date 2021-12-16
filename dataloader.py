import math
import random
from glob import glob


import torch
import numpy as np
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader


def get_dataloader(opt, mode):
    dataset = X4K1000FPS if opt.model == 'mod' else X4KLIIF
    if mode == 'train':
        train_dataset = dataset(f'{opt.data_root}/train', opt.num_frames, opt.patch_size)
        val_dataset = dataset(f'{opt.data_root}/val', opt.num_frames, opt.patch_size, False)
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False,
                                      num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        return train_dataloader, val_dataloader

    elif mode == 'test':
        test_dataset = None
        return test_dataset

    else:
        exit(f"Unsupported exp mode {mode}")


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

        return frames, target_frame, target_t

    def mod_item(self, idx):
        pass

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
            frames, target_frame, target_t = self.augment(frames, target_frame, target_t)

        else:
            frames = np.stack(frames + [target_frame], axis=0)

            # # Patchify
            # h, w, c = target_frame.shape
            # ix = random.randrange(0, w - self.patch_size + 1)
            # iy = random.randrange(0, h - self.patch_size + 1)
            # frames = frames[:, iy:iy + self.patch_size, ix:ix + self.patch_size, :]

        frames = frames.transpose((3, 0, 1, 2)) / 127.5 - 1
        frames = torch.Tensor(frames.astype(float))

        return frames[:, :-1, :, :], frames[:, -1, :, :], target_t


class X4KLIIF(Dataset):
    def __init__(self, data_root, num_frames, patch_size, is_train=True, scale_min=1, scale_max=4):
        super(X4KLIIF, self).__init__()
        self.is_train = is_train
        self.scale_min = scale_min
        self.scale_max = scale_max
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

        return frames, target_frame, target_t

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


        scale = random.uniform(self.scale_min, self.scale_max)

        # Read frames
        frames = []
        for idx in selected_idx:
            cur_f = Image.open(frame_paths[idx]).convert('RGB')
            lr_res = math.floor(cur_f.size[0] / scale + 1e-9)
            cur_f = cur_f.resize((lr_res, lr_res), Image.BICUBIC)
            frames.append(np.array(cur_f))

        return frames



if __name__ == '__main__':
    from config import Config
    config = Config()
    t, v = get_dataloader(config, config.mode)

    for d in t:
        input_frames = d
        print(input_frames.shape)
        break

    for d in v:
        input_frames, target_frames, t = d
        print(input_frames.shape)
        break

