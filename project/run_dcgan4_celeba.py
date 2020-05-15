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
    e = utils.create_experiment("experiments/dcgan4_celeba")

    # Hyperparameters
    e.params["shuffle"]     = True # Shuffle the contents in the dataset
    e.params["num_workers"] = 4    # Number of worker threads for dataloader
    e.params["batch_size"]  = 128  # Size of one batch during training
    e.params["nc"]          = 3    # Number of channels in the training images (color RGB uses 3 channels)
    e.params["nz"]          = 100  # Size of z latent vector (i.e. size of generator input)
    e.params["im_size"]     = 64   # Size of the images discriminated and generated.
    e.params["num_epochs"]  = e.input_int("number of epochs", 5) # Number of epochs 
    e.params["lr"]          = 0.0002       # Learning rate for optimizer
    e.params["betas"]       = (0.5, 0.999) # Betas hyperparameter for Adam optimizers
    e.params["patience"]    = 7 # Number of epochs to wait before early stopping

    # Setup the CIFAR10 dataset
    transform = utils.image_transform(e.params["im_size"])
    data_dir = "data/celeba/"

    # WARNING DOWNLOAD IS 1.4 GB in size!!! 
    # train_dataset = datasets.CelebA(data_dir, split="train", download=False, transform=transform)

    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    e.setup_dataloader((train_dataset, None, None))
    
    # Plot a subset of the training dataset
    utils.plot_data_subset(e.fname("dataset_image.png"), train_dataset, show_labels=False)

    # Setup the two models
    e.generator = models.dcgan4_generator(e)
    e.discriminator = models.dcgan4_discriminator(e)

    # Criterion (or loss function) used
    e.criterion = nn.BCELoss()

    # The optimizer for weight updating
    e.g_optimizer = optim.Adam(e.generator.parameters(), lr=e.params["lr"], betas=e.params["betas"])
    e.d_optimizer = optim.Adam(e.discriminator.parameters(), lr=e.params["lr"], betas=e.params["betas"])

    # Train model and plot results
    gantraining.train_model(e)
    gantraining.plot_all(e)
