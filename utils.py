import PIL.Image as pil_image
import numpy as np
import torch


def load_image(path):
    return pil_image.open(path).convert('RGB')


def generate_lr(image, scale):
    image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
    return image


def modcrop(image, modulo):
    w = image.width - image.width % modulo
    h = image.height - image.height % modulo
    return image.crop((0, 0, w, h))


def generate_patch(image, patch_size, stride):
    for i in range(0, image.height - patch_size + 1, stride):
        for j in range(0, image.width - patch_size + 1, stride):
            yield image.crop((j, i, j + patch_size, i + patch_size))


def image_to_array(image):
    return np.array(image).transpose((2, 0, 1))


def normalize(x):
    return x / 255.0


def denormalize(x):
    if type(x) == torch.Tensor:
        return (x * 255.0).clamp(0.0, 255.0)
    elif type(x) == np.ndarray:
        return (x * 255.0).clip(0.0, 255.0)
    else:
        raise Exception('The denormalize function supports torch.Tensor or np.ndarray types.', type(x))


def rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def PSNR(a, b, max=255.0, shave_border=0):
    assert type(a) == type(b)
    assert (type(a) == torch.Tensor) or (type(a) == np.ndarray)

    a = a[shave_border:a.shape[0]-shave_border, shave_border:a.shape[1]-shave_border]
    b = b[shave_border:b.shape[0]-shave_border, shave_border:b.shape[1]-shave_border]

    if type(a) == torch.Tensor:
        return 10. * ((max ** 2) / ((a - b) ** 2).mean()).log10()
    elif type(a) == np.ndarray:
        return 10. * np.log10((max ** 2) / np.mean(((a - b) ** 2)))
    else:
        raise Exception('The PSNR function supports torch.Tensor or np.ndarray types.', type(a))


def load_weights(model, path):
    state_dict = model.state_dict()
    for n, p in torch.load(path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    return model


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
