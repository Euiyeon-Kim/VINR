import random

import torch
import numpy as np


def make_coord(shape, ranges=None, flatten=True):
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


def get_flat_coord_rgb_from_img(img):
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def make_liif_data(target_frame):
    # target_frame: H, W, C
    target_frame = target_frame.transpose((2, 0, 1)) / 127.5 - 1
    target_frame = torch.Tensor(target_frame.astype(float))
    target_coord, target_rgb = get_flat_coord_rgb_from_img(target_frame.contiguous())

    cell = torch.ones_like(target_coord)
    cell[:, 0] *= 2 / target_frame.shape[-2]
    cell[:, 1] *= 2 / target_frame.shape[-1]
    return target_coord, target_rgb, cell


def norm_img_arr_and_to_tensor(img_arr):
    # T, H, W, C -> C, T, H, W
    img_arr = img_arr.transpose((3, 0, 1, 2)) / 127.5 - 1
    img_arr = torch.Tensor(img_arr.astype(float))
    return img_arr


def augment(frames, target_frame, target_t, patch_size, zero_centered=False):
        # Reverse
        if random.random() < 0.5:
            target_t = -target_t if zero_centered else 1 - target_t
            frames.reverse()

        frames = np.stack(frames + [target_frame], axis=0)

        h, w, c = target_frame.shape
        ix = random.randrange(0, w - patch_size + 1)
        iy = random.randrange(0, h - patch_size + 1)
        frames = frames[:, iy:iy + patch_size, ix:ix + patch_size, :]

        # Flip
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :]

        # Rotate
        rot = random.randint(0, 3)
        frames = np.rot90(frames, rot, (1, 2))

        return frames, target_t


def sample_t(num_frames):
    total_frame = 65

    # Sample input frames and target frame
    max_td = int((total_frame - 1) / (num_frames - 1))
    td = random.randint(2, max_td)
    first_frame_idx = random.randint(0, (total_frame - 1) - ((num_frames - 1) * td))
    last_frame_idx = first_frame_idx + ((num_frames - 1) * td)
    selected_idx = np.linspace(first_frame_idx, last_frame_idx, num_frames).astype(int)
    target_idx = random.randint(first_frame_idx, last_frame_idx)
    return selected_idx, target_idx


def normalize_ts(selected_idxs, target_idx, zero_centered=False):
    if zero_centered:
        assert len(selected_idxs) % 2 == 1, "To use zero-centered t, # input frame should be odd"
        sub_idx = int(len(selected_idxs)/2)
    else:
        sub_idx = 0
    target_t = (target_idx - selected_idxs[sub_idx]) / (selected_idxs[-1] - selected_idxs[sub_idx])
    return target_t

