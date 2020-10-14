import numpy as np
from tifffile import imsave


def save_stn(name, img):
    assert len(img.shape) == 3
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    imsave(name, img)


def save_tensor(name, data):
    data = data[0, 0].cpu().numpy()
    imsave(name, data)
