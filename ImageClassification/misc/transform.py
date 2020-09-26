from __future__ import division
import numpy as np
import torch


__all__ = ["RandomLighting"]


class RandomLighting(object):
    def __init__(self, alpha, eigval=None, eigvec=None):
        self.alphastd = alpha
        self.eigval = eigval
        self.eigvec = eigvec
    def __call__(self, img):
        if self.alphastd <= 0:
            return img

        if self.eigval is None:
            eigval = np.array([55.46, 4.794, 1.148])
        if self.eigvec is None:
            eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                               [-0.5808, -0.0045, -0.8140],
                               [-0.5836, -0.6948, 0.4203]])

        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.dot(eigvec * alpha, eigval)
        # img += nd.array(rgb)
        img += torch.from_numpy(rgb).to(torch.float).div(255).view(3,1,1)
        return img




if __name__ == "__main__":
    x = np.random.randn(3, 1, 1)
    tx = torch.from_numpy(x)
    nx = nd.array(x.reshape([1,1,3]))



    trans = RandomLighting(0.1)
    # trans = transforms.RandomLighting(0.1)
    out = trans(nx)

    print(x)
    print(nx)