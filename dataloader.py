import random
from glob import glob

# import cv2
import torch
from cv2 import cv2
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader


def get_dataloader(opt, mode):
    if mode == 'train':
        train_dataset = X4K1000FPS(f'{opt.data_root}/train', opt.num_frames, opt.patch_size)
        val_dataset = X4K1000FPS(f'{opt.data_root}/val', opt.num_frames, opt.patch_size, False)
    elif mode == 'debug':
        train_dataset = X4K1000FPS(f'{opt.data_root}/toy', opt.num_frames, opt.patch_size)
        val_dataset = X4K1000FPS(f'{opt.data_root}/toy', opt.num_frames, opt.patch_size, False)
    else:
        exit(f"Unsupported exp mode {mode}")

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False,
                                  num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=opt.num_workers)

    return train_dataloader, val_dataloader


class X4K1000FPS(Dataset):
    def __init__(self, data_root, num_frames, patch_size, is_train=True):
        super(X4K1000FPS, self).__init__()
        self.is_train = is_train
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.clips = glob(f'{data_root}/*/*')

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        cur_clip = self.clips[item]
        frame_paths = natsorted(glob(f'{cur_clip}/*.png'))
        assert len(frame_paths) == 65, f'Dataset is not complete. Check {cur_clip}'

        # Set options
        td = random.randint(2, int(64 / (self.num_frames - 1)))         # max_td = int(64 / (self.num_frames - 1))
        first_frame_idx = random.randint(0, 64 - ((self.num_frames - 1) * td))
        last_frame_idx = first_frame_idx + ((self.num_frames - 1) * td)
        selected_idx = np.linspace(first_frame_idx, last_frame_idx, self.num_frames).astype(int)
        target_idx = random.randint(first_frame_idx, last_frame_idx)

        # Read frames
        frames = []
        for idx in selected_idx:
            frames.append(cv2.imread(frame_paths[idx]))
        target_frame = cv2.imread(frame_paths[target_idx])
        target_t = (target_idx - first_frame_idx) / (last_frame_idx - first_frame_idx)

        if self.is_train:
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

        else:
            frames = np.stack(frames + [target_frame], axis=0)

            # Patchify
            h, w, c = target_frame.shape
            ix = random.randrange(0, w - self.patch_size + 1)
            iy = random.randrange(0, h - self.patch_size + 1)
            frames = frames[:, iy:iy + self.patch_size, ix:ix + self.patch_size, :]

        frames = frames.transpose((3, 0, 1, 2)) / 127.5 - 1
        frames = torch.Tensor(frames.astype(float))

        return frames[:, :-1, :, :], frames[:, -1, :, :], target_t


if __name__ == '__main__':
    from config import Config
    config = Config()
    train_dataloader = get_dataloader(config, 'train')
    for d in train_dataloader:
        input_frames, target_frames, t = d
        exit()


