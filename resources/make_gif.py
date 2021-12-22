from glob import glob
from PIL import Image
from natsort import natsorted
import imageio

EXP_NAME = 'frame5_mod_z128'

if __name__ == '__main__':
    clips = natsorted(glob(f'exps/{EXP_NAME}/infer/*/*'))
    for c in clips:
        print(c)
        frame_paths = natsorted(glob(f'{c}/*'))
        frames = [Image.open(i) for i in frame_paths]
        imageio.mimsave(f'{c}.gif', frames, fps=8000)
