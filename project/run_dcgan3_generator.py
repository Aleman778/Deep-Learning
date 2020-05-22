"""Run your trained DCGAN3 model and experiment with the generator.
"""


import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim

import utils
import models
import gantraining


def plot_images(image_tensor, filename):
    image = vutils.make_grid(image_tensor, padding=2, nrow=10, normalize=True).detach().cpu()
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.savefig(e.fname(filename), bbox_inches="tight", pad_inches=0.2)

    
if __name__ == '__main__':
    matplotlib.rcParams["image.interpolation"] = 'nearest'
    # Setup a new experiment, keeps results in one folder
    e = utils.create_experiment("experiments/dcgan3_cifar10")

    # Hyperparameters
    e.params["shuffle"]     = True # Shuffle the contents in the dataset
    e.params["num_workers"] = 4    # Number of worker threads for dataloader
    e.params["batch_size"]  = 128  # Size of one batch during training
    e.params["nc"]          = 3    # Number of channels in the training images (color RGB uses 3 channels)
    e.params["nz"]          = 100  # Size of z latent vector (i.e. size of generator input)
    e.params["im_size"]     = 32   # Size of the images discriminated and generated.
    e.params["num_epochs"]  = 0 # Number of epochs 
    e.params["lr"]          = 0.0002       # Learning rate for optimizer
    e.params["betas"]       = (0.5, 0.999) # Betas hyperparameter for Adam optimizers
    e.params["patience"]    = 7 # Number of epochs to wait before early stopping

    # Setup the two models
    e.generator = models.dcgan3_generator(e)
    e.discriminator = models.dcgan3_discriminator(e)

    # Criterion (or loss function) used
    e.criterion = nn.BCELoss()

    # The optimizer for weight updating
    e.g_optimizer = optim.Adam(e.generator.parameters(), lr=e.params["lr"], betas=e.params["betas"])
    e.d_optimizer = optim.Adam(e.discriminator.parameters(), lr=e.params["lr"], betas=e.params["betas"])

    # Load a pretrained model
    gantraining.load_checkpoint(e, "experiments/dcgan3_cifar10/train_80_epochs/checkpoint.pth")

    # Let's create 100 images from random noise
    for figid in tqdm(range(10)):
        inputs = torch.randn(100, e.params["nz"], 1, 1, device=e.device)
        outputs = e.generator(inputs)
        plot_images(outputs, "generated_from_random_noise_" + str(figid) + ".png")

    for figid in tqdm(range(10)):
        inputs = torch.zeros(100, e.params["nz"], 1, 1, dtype=torch.float)
        for i in range(10):
            for j in range(10):
                inputs[i*10 + j][figid] = float(i)/10.0       
                inputs[i*10 + j][figid + 1] = float(j)/10.0   
        inputs = inputs.to(e.device)
        outputs = e.generator(inputs)
        plot_images(outputs, "generated_" + str(figid) + ".png")
