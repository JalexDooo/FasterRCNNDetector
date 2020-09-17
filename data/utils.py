import numpy as np
from PIL import Image
import random
import torch as t

from skimage import transform as sktsf
from torchvision import transforms as tvtsf


def read_image(path, dtype=np.float32, color=True):
    """
    Read an image from a file.

    :param path: A path of image file.
    :param dtype: The type of array. The default type is float32.
    :param color: 3 channels or 1 channels. The default is 3 channels.
    :return: image array.
    """
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        return img[np.newaxis]
    else:
        return img.transpose((2, 0, 1))


def normalize(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img


def preprocess(img, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size/min(H, W)
    scale2 = max_size/max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)
    img = normalize(img)
    return img


def resize_box(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / float(in_size[0])
    x_scale = float(out_size[1]) / float(in_size[1])
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def random_flip(img, y_random=False, x_random=False, return_param=False, copy=False):
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = t.Tensor(img.numpy()[:, ::-1, :].copy())
    if x_flip:
        img = t.Tensor(img.numpy()[:, :, ::-1].copy())

    if copy:
        img = img.copy()
    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox
