"""Recreated experiment from pytorch-mnist-GAN.ipynb file.
This runs a 4-layer fully connected Generative Adversarial Network
on the MNIST dataset.
"""

import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn, optim
from torchvision import datasets

import utils
import models
import gantraining

def plot_images(image_tensor, filename):
    image = vutils.make_grid(image_tensor, padding=2, nrow=15, normalize=True).detach().cpu()
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.savefig(e.fname(filename), bbox_inches="tight", pad_inches=0.2)


if __name__ == '__main__':
    # Setup a new experiment, keeps results in one folder
    e = utils.create_experiment("experiments/densegan4_mnist")

    # Hyperparameters
    e.params["shuffle"]     = True # Shuffle the contents in the dataset
    e.params["num_workers"] = 4    # Number of worker threads for dataloader
    e.params["batch_size"]  = 128  # Size of one batch during training
    e.params["nc"]          = 1    # Number of channels in the training images (color RGB uses 3 channels)
    e.params["nz"]          = 100  # Size of z latent vector (i.e. size of generator input)
    e.params["ndf"]         = 1024 # Number of features in the discriminator network.
    e.params["ngf"]         = 256  # Number of features in the generator network.
    e.params["im_size"]     = 28   # Size of the images discriminated and generated.
    e.params["num_epochs"]  = e.input_int("number of epochs", 5) # Number of epochs 
    e.params["lr"]          = 0.0002       # Learning rate for optimizer
    e.params["betas"]       = (0.5, 0.999) # Betas hyperparameter for Adam optimizers
    e.params["patience"]    = 50 # Number of epochs to wait before early stopping

    # Setup the MNIST dataset
    transform = utils.image_transform_grayscale(e.params["im_size"])
    data_dir = "data/mnist/"
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transform)
    e.setup_dataloader((train_dataset, None, test_dataset))
    
    # Plot a subset of the training dataset
    utils.plot_data_subset(e.fname("dataset_image.png"), train_dataset)

    # Setup the two models
    e.generator = models.densegan4_generator(e)
    e.discriminator = models.densegan4_discriminator(e)

    # Criterion (or loss function) used
    e.criterion = nn.BCELoss()

    # The optimizer for weight updating
    e.g_optimizer = optim.Adam(e.generator.parameters(), lr=e.params["lr"], betas=e.params["betas"])
    e.d_optimizer = optim.Adam(e.discriminator.parameters(), lr=e.params["lr"], betas=e.params["betas"])

    # Load a pretrained model
    gantraining.load_checkpoint(e, "experiments/densegan4_mnist/train_all_50_epochs/checkpoint.pth")

    for figid in tqdm(range(10)):
        inputs = torch.randn((15*15, e.params["nz"]), device=e.device)
        print(inputs.size())
        outputs = e.generator(inputs).view(15*15, 1, 28, 28)
        print(outputs.size())
        plot_images(outputs, "generated_" + str(figid) + ".png")
