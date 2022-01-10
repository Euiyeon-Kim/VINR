import torch
from torchvision.utils import save_image


def save_rgbtensor(rgbtensor, path, norm=True):
    if norm:
        rgbtensor = (rgbtensor + 1.) / 2.
    save_image(rgbtensor, path)


def psnr(gt, pred, norm=True):
    if norm:    # (-1, 1) to (0, 255)
        gt = ((gt + 1.) / 2.) * 255.
        pred = ((pred + 1.) / 2.) * 255.
    diff = gt - pred
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    if rmse == 0:
        return float('inf')
    return 20 * torch.log10(255. / rmse)