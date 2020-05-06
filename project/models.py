import torch
from torch import nn


"""Different model implementations that can be used.
"""


def dcgan3_discriminator(nc=3, im_size=32):
    """Discriminator model for GANs using CNN network. This is based on the dcgan paper
    [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) except it uses 3 conv layers instead of 4.
    This model should work perfectly images of size 32 or smaller.

    nc - the number of components in the image (3 for RGB, 1 for grayscale)

    im_size - the size of the input image in pixels"""

    return nn.Sequential(
        # INPUT - image of dimensions (nc) * im_size * im_size
        nn.Conv2d(nc, im_size, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV1 - current dimension is: (im_size*2) * (im_size/2) * (im_size/2)
        nn.Conv2d(im_size, im_size*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*2),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV2 - current dimension is: (im_size*4) * (im_size/4) * (im_size/4)
        nn.Conv2d(im_size*2, im_size*4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*4),
        nn.LeakyReLU(0.2, inplace=True),

        # CONV3 - current dimension is: (im_size*8) * (im_size/8) * (im_size/8)
        nn.Conv2d(im_size*4, 1, kernel_size=4, stride=1, padding=0, bias=False),
        nn.Sigmoid()

        # OUTPUT - single number between 0.0 (fake) and 1.0 (real)
    )


def dcgan3_generator(nc=3, im_size=32):
    """Generator model for GANs using CNN network. This is based on the dcgan paper
    [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) except it uses 3 conv layers instead of 4.
    This model should generate images of size 32 or smaller.

    nc - the number of components in the image (3 for RGB, 1 for grayscale)

    im_size - the size of the output image in pixels"""

    return nn.Sequential(
        # INPUT - latent vector Z is the input to convolution
        nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(inplace=True),

        # CONV1 - current dimension is: (ngf) * (image_size/8) * (image_size/8)
        nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(inplace=True),

        # CONV2 - current dimension is: (ngf*4) * (image_size/4) * (image_size/4)
        nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(inplace=True),

        # CONV3 - current dimension is: (ngf) * (image_size/2) * (image_size/2)
        nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()

        # OUTPUT - dimension is: (nc) * image_size * image_size
    )
    
