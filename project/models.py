import torch
from torch import nn


"""Different model implementations that can be used.
"""


def _init_weights(model):
    """From the DCGAN paper the authors specify that all model weights shall 
    be randomly initialized from a normal distribution with mean=0 and stdev=0.02.
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
            


def dcgan3_discriminator(e):
    """Discriminator model for GANs using CNN network. This is based on the dcgan paper
    [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) except it uses 3 conv layers instead of 4.
    This model should work perfectly images of size 32 or smaller.

    e - the experiment instance
    nc - the number of components in the image (3 for RGB, 1 for grayscale)
    im_size - the size of the input image in pixels"""
    nc = e.params["nc"] 
    im_size = e.params["im_size"]

    model = nn.Sequential(
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
    ).to(e.device)
    model.apply(_init_weights)
    return model


def dcgan3_generator(e):
    """Generator model for GANs using CNN network. This is based on the dcgan paper
    [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) except it uses 3 conv layers instead of 4.
    This model should generate images of size 32 or smaller.

    e - the experiment instance with params set:
    - nz - the size of the latent vector (input noise to generator)
    - nc - the number of components in the image (3 for RGB, 1 for grayscale)
    - im_size - the size of the output image in pixels"""

    nz = e.params["nz"] 
    nc = e.params["nc"] 
    im_size = e.params["im_size"]

    model = nn.Sequential(
        # INPUT - latent vector Z is the input to convolution
        nn.ConvTranspose2d(nz, im_size*4, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(im_size*4),
        nn.ReLU(inplace=True),

        # CONV1 - current dimension is: (im_size) * (im_size/8) * (im_size/8)
        nn.ConvTranspose2d(im_size*4, im_size*2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size*2),
        nn.ReLU(inplace=True),

        # CONV2 - current dimension is: (im_size*4) * (im_size/4) * (im_size/4)
        nn.ConvTranspose2d(im_size*2, im_size, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(im_size),
        nn.ReLU(inplace=True),

        # CONV3 - current dimension is: (im_size) * (im_size/2) * (im_size/2)
        nn.ConvTranspose2d(im_size, nc, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()

        # OUTPUT - dimension is: (nc) * im_size * im_size
    ).to(e.device)
    model.apply(_init_weights)
    return model
    
