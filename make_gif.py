from glob import glob
from PIL import Image
from natsort import natsorted
import imageio


if __name__ == '__main__':
    prefix = 'sub'
    clips = natsorted(glob(f'tmp/*'))
    for c in clips:
        frame_paths = natsorted(glob(f'{c}/*'))
        frames = []
        frames = [Image.open(i) for i in frame_paths]
        name = c.split('/')[-1]
        print(len(frames), name)
        imageio.mimsave(f'./{prefix}_{name}.gif', frames, fps=30)

