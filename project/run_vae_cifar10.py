"""Recreated experiment from gan_experiment.py file.
This runs a 3 convolutional layer Generative Adversarial Network
on the CIFAR10 dataset.
"""

import torch
import torchvision
from torch import nn, optim
from torchvision import datasets

import utils
import models
import gantraining


if __name__ == '__main__':
    # Setup a new experiment, keeps results in one folder
    e = utils.create_experiment("experiments/dcgan3_cifar10")

    # Hyperparameters
    e.params["shuffle"]     = True # Shuffle the contents in the dataset
    e.params["num_workers"] = 4    # Number of worker threads for dataloader
    e.params["batch_size"]  = 128  # Size of one batch during training
    e.params["nc"]          = 3    # Number of channels in the training images (color RGB uses 3 channels)
    e.params["nz"]          = 100  # Size of z latent vector (i.e. size of generator input)
    e.params["nef"]         = 64   # Number of features in the discriminator network.
    e.params["ndf"]         = 64   # Number of features in the generator network.
    e.params["mean"]        = 0.0  # Mean value of 
    e.params["stdev"]       = 1.0  # 
    e.params["im_size"]     = 32   # Size of the images discriminated and generated.
    e.params["num_epochs"]  = e.input_int("number of epochs", 5) # Number of epochs 
    e.params["lr"]          = 0.0002       # Learning rate for optimizer
    e.params["betas"]       = (0.5, 0.999) # Betas hyperparameter for Adam optimizers
    e.params["patience"]    = 7 # Number of epochs to wait before early stopping

    # Setup the CIFAR10 dataset
    transform = utils.image_transform(e.params["im_size"])
    data_dir = "data/cifar10/"
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform)
    e.setup_dataloader((train_dataset, None, test_dataset))
    
    # Plot a subset of the training dataset
    utils.plot_data_subset(e.fname("dataset_image.png"), train_dataset)

    # Setup the two models
    e.encoder = models.VAE_Encoder(e)
    
    # Criterion (or loss function) used
    e.criterion = nn.BCELoss()

    # The optimizer for weight updating
    e.g_optimizer = optim.Adam(e.generator.parameters(), lr=e.params["lr"], betas=e.params["betas"])
    e.d_optimizer = optim.Adam(e.discriminator.parameters(), lr=e.params["lr"], betas=e.params["betas"])

    # Train model and plot results

    e.encoder(next(iter(e.train_loader)))