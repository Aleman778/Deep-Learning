"""Recreated experiment from gan_experiment.py file.
This runs a 3 convolutional layer Generative Adversarial Network
on the CIFAR10 dataset.
"""

import torchvision
from torchvision import datasets

import utils
import models
import gantraining


def run_experiment(experiment):
    # Setup CUDA device if available
    device = utils.cuda_device_if_available();

    # Optionally load a pretrained model
    pretrained = False
    pretrained_path = "models/best_model.pth.tar"

    # Hyperparameters
    shuffle = True   # Shuffle the contents in the dataset
    num_workers = 4  # Number of worker threads for dataloader
    image_size = 32  # Size of input images (images are resized to this using transformer)
    batch_size = 128 # Size of one batch during training
    nc = 3           # Number of channels in the training images (color RGB uses 3 channels)
    nz = 100         # Size of z latent vector (i.e. size of generator input)
    im_size = 32     # Size of the images discriminated and generated.
    num_epochs = 5   # Number of epochs
    lr = 0.0002      # Learning rate for optimizer
    beta1 = 0.5      # Beta1 hyperparameter for Adam optimizers

    # Setup the two models
    generator = models.dcgan3_generator(nc, im_size)
    discriminator = models.dcgan3_discriminator(nc, im_size)

    # Setup the dataset
    data_dir = "data/cifar10/"
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    # Plot the dataset
    utils.plot_dataset(experiment.fname("dataset_image.png"), num_sampels=16, show_labels=False)


    # Done with experiment
    experiment.close()
    

if __name__ == '__main__':
    utils.create_experiment("experiments/dcgan3", run_experiment)
