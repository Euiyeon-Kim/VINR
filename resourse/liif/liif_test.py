import os
from glob import glob
from natsort import natsorted

import numpy as np
from PIL import Image
import torch
from torch import nn

from models.liif import XVFIEncoder, LIIF, ModRGBMapper, VINR
from resourse.mod.mod_trainer import save_rgbtensor


if __name__ == '__main__':
    from test_config import Config
    opt = Config()
    os.makedirs(f'{opt.exp_dir}/infer', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = XVFIEncoder(in_c=3, num_frames=opt.num_frames, nf=opt.z_dim, n_blocks=2)
    liif = LIIF(opt.z_dim)
    mapper = ModRGBMapper(out_dim=3)
    model = VINR(encoder, liif, mapper)
    model = nn.DataParallel(model)

    ckpt = torch.load(f'{opt.exp_dir}/ckpt/best.pth')
    model.load_state_dict(ckpt['model'], strict=False)

    # import matplotlib.pyplot as plt
    # lff_mag = sorted(model.mapper.lff.ffm.layer.weight.detach().numpy()[:, 0])
    # lff_phase = sorted(model.mapper.lff.ffm.layer.bias.detach().numpy())
    # plt.hist(lff_mag, bins=64, label='magnitude')
    # plt.hist(lff_phase, bins=64, label='phase')
    # plt.legend()
    # plt.savefig('lff.png')

    clips = natsorted(glob(f'{opt.data_root}/val/*/*'))
    total_f = 33
    target_t = np.expand_dims(np.arange(total_f)/32, -1)
    target_t = torch.from_numpy(target_t).float()

    for clip in clips:
        clip_name = clip.split('/')[-1]
        os.makedirs(f'{opt.exp_dir}/infer/{clip_name}', exist_ok=True)
        inps = natsorted(glob(f'{clip}/*'))
        frames = []
        for i in range(0, len(inps), 8):
            frames.append(np.array(Image.open(inps[i]).convert('RGB')))
        Image.fromarray(frames[0]).save(f'{opt.exp_dir}/infer/{clip_name}/0.png')
        Image.fromarray(frames[-1]).save(f'{opt.exp_dir}/infer/{clip_name}/1.png')

        frames = np.stack(frames, axis=0)
        frames = frames.transpose((3, 0, 1, 2)) / 127.5 - 1
        inp_frames = torch.unsqueeze(torch.Tensor(frames.astype(float)).to(device), 0)

        for t in target_t:
            pred_frame = model(inp_frames, t)
            save_rgbtensor(pred_frame[0], f'{opt.exp_dir}/infer/{clip_name}/{t.item():04f}.png')
