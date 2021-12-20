import os
import argparse
from glob import glob

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from common_model import Encoder
from mod_model import Modulator, ModRGBMapper, VINR
from dataloader import X4K1000FPS
from trainer import save_rgbtensor


if __name__ == '__main__':
    from test_config import Config
    opt = Config()

    os.makedirs(f'{opt.exp_dir}/infer')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(in_dim=3*opt.num_frames, out_dim=opt.z_dim)
    modulator = Modulator(in_f=opt.z_dim, hidden_node=256, depth=4)
    mapper = ModRGBMapper(out_dim=3, hidden_node=256, depth=5)
    model = VINR(encoder, modulator, mapper)
    model = nn.DataParallel(model).to(device)

    ckpt = torch.load(f'{opt.exp_dir}/ckpt/best.pth')
    model.load_state_dict(ckpt['model'], strict=False)

    # import matplotlib.pyplot as plt
    # lff_mag = sorted(model.mapper.lff.ffm.layer.weight.detach().numpy()[:, 0])
    # lff_phase = sorted(model.mapper.lff.ffm.layer.bias.detach().numpy())
    # plt.hist(lff_mag, bins=64, label='magnitude')
    # plt.hist(lff_phase, bins=64, label='phase')
    # plt.legend()
    # plt.savefig('lff.png')

    clips = glob(f'{opt.data_root}/test/*/*')
    total_f = opt.num_frames * 8
    target_t = np.linspace((1 / total_f), (1 - (1 / total_f)), (total_f - 1)).reshape((total_f-1, 1))
    target_t = torch.from_numpy(target_t).float()

    for clip in clips:
        clip_name = clip.split('/')[-1]
        inps = glob(f'{clip}/*')
        for i in range(0, len(inps) - opt.num_frames+1, opt.num_frames-1):
            frames = []
            for j in range(opt.num_frames):
                frames.append(np.array(Image.open(inps[i+j]).convert('RGB')))
            frames = np.stack(frames, axis=0)
            frames = frames.transpose((3, 0, 1, 2)) / 127.5 - 1
            inp_frames = torch.unsqueeze(torch.Tensor(frames.astype(float)).to(device), 0)
            for t in target_t:
                pred_frame = model(inp_frames, t)
                save_rgbtensor(pred_frame[0], f'{opt.exp_dir}/infer/{clip_name}_{i}{j}_{target_t[0]}.png')

        # input_frames, target_frame, target_t = data
        #
        # input_frames = input_frames.to(device)

        # pred_frame = model(input_frames, target_t)


