import random
from glob import glob

import numpy as np
from PIL import Image
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader


def get_dataloader(opt, mode):
    dataset = X4K1000FPS(f'{opt.data_root}/{mode}', opt.num_frames)
    shuffle = True if mode == 'train' else False
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle, drop_last=False)
    return dataloader


class X4K1000FPS(Dataset):
    def __init__(self, data_root, num_frames):
        super(X4K1000FPS, self).__init__()
        self.num_frames = num_frames
        self.clips = glob(f'{data_root}/*/*')

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        cur_clip = self.clips[item]
        frames = natsorted(glob(f'{cur_clip}/*.png'))
        assert len(frames) == 65, f'Dataset is not complete. Check {cur_clip}'

        # max_td = int(64 / (self.num_frames - 1))
        td = random.randint(2, int(64 / (self.num_frames - 1)))
        first_frame_idx = random.randint(0, 64 - ((self.num_frames - 1) * td))
        last_frame_idx = first_frame_idx + ((self.num_frames - 1) * td)
        selected_idx = np.linspace(first_frame_idx, last_frame_idx, self.num_frames).astype(int)
        target_idx = random.randint(first_frame_idx, last_frame_idx)

        frames = []
        for idx in selected_idx:
            frames.append(Image.open(frames[idx]))

        return



if __name__ == '__main__':
    from config import Config
    config = Config()
    train_dataloader = get_dataloader(config, 'train')
    for d in train_dataloader:
        print(d)
        exit()


