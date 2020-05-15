"""Generative Adversarial Network training code is slightly modified from 
[DCGAN PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)"""

import logging
import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from torch import nn


import training
import models

        
##############################################################
# Heler functions
##############################################################


def save_checkpoint(e, epoch, iteration):
    """Stores GAN training checkpoint. Can be restored via load_checkpoint."""
    data = {"epoch": epoch,
            "iteration": iteration,
            "g_state_dict": e.generator.state_dict(),
            "d_state_dict": e.discriminator.state_dict(),
            "g_optimizer_state_dict": e.g_optimizer.state_dict(),
            "d_optimizer_state_dict": e.d_optimizer.state_dict()}
    training._save_checkpoint(e, data)


def load_checkpoint(e, path):
    """Loads the checkpoint """
    data = training._load_checkpoint(e, path)
    e.generator.load_state_dict(data["g_state_dict"])
    e.discriminator.load_state_dict(data["d_state_dict"])
    e.g_optimizer.load_state_dict(data["g_optimizer_state_dict"])
    e.d_optimizer.load_state_dict(data["d_optimizer_state_dict"])
    return (data["epoch"], data["iteration"])


##############################################################
# Training
##############################################################


def train_model(e, pretrained=False, flatten_input=False):
    """Trains the model current loaded in the experiment."""
    # Labels is defined as follows
    fake_label = 0
    real_label = 1

    # Basic dataset information
    train_size = len(e.train_loader)

    # Generate some noise, used to see generator progression.
    fixed_noise = torch.randn(64, e.params["nz"], 1, 1, device=e.device)

    # Keep track of some running statistics to view after training
    if not pretrained:
        e.training["image_list"] = []
        e.training["g_batch_losses"] = []
        e.training["d_batch_losses"] = []
        e.training["g_epoch_losses"] = []
        e.training["d_epoch_losses"] = []
        e.training["g_epoch_accuracies"] = []
        e.training["d_epoch_accuracies"] = []

    ##############################################################
    # Training loop
    ##############################################################

    # Iteration is the number of batches trained.
    iteration = 0

    for epoch in range(e.params["num_epochs"]):
        epoch_stats = {
            "d_train_loss": 0.0,
            "d_train_corrects": 0.0,
            "g_train_loss": 0.0,
            "g_train_corrects": 0.0}

        for batch_num, (real_images, _) in enumerate(e.train_loader):
            if flatten_input:
                real_images = real_images.view(-1, e.params["im_size"]**2)

            ##############################################################
            # Training Discriminator with REAL (dataset) images
            ##############################################################


            # Reset the discriminator gradients
            e.discriminator.zero_grad()

            # Setup the input data and label
            real_images = real_images.to(e.device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), real_label, device=e.device)

            # Forward pass real batch through discriminator
            output = e.discriminator(real_images).view(-1)

            # Define prediction as the rounded value from output
            d_pred_real = output.round()

            # Compute the loss using set criterion
            d_loss_real = e.criterion(output, label)

            # Calculate the gradients for discriminator in backward pass
            d_loss_real.backward()
            d_x = output.mean().item()


            ##############################################################
            # Training Discriminator with FAKE (generated) images
            ##############################################################


            # Generate batch of latent vectors 
            noise = torch.randn(b_size, e.params["nz"], 1, 1, device=e.device)
            if flatten_input:
                noise = noise.view(-1, e.params["nz"])

            # Generate fake image batch using our generator
            fake_images = e.generator(noise)
            label.fill_(fake_label)

            # Make predictions for the batch of fake images using discriminator
            output = e.discriminator(fake_images.detach()).view(-1)

            # Define prediction as the rounded value from output
            d_pred_fake = output.round()

            # Compute the loss using set criterion
            d_loss_fake = e.criterion(output, label)

            # Calculate the gradients for discriminator in backward pass
            d_loss_fake.backward()
            d_g_z1 = output.mean().item()

            # Now update discriminator
            e.d_optimizer.step()


            # Update statistics
            d_batch_loss = d_loss_real.item()
            d_batch_loss += d_loss_fake.item()
            d_corrects = torch.sum(d_pred_real == 1).item()
            d_corrects += torch.sum(d_pred_fake == 0).item()
            epoch_stats["d_train_loss"] += d_batch_loss
            epoch_stats["d_train_corrects"] += d_corrects


            ##############################################################
            # Training Generator
            ##############################################################


            # Reset the discriminator gradients
            e.generator.zero_grad()

            # Since we just updated our discriminator we now want to perform
            # another forward pass of the batch of fake images trough our discriminator
            output = e.discriminator(fake_images).view(-1)

            # When training the generator we want the discriminator to think our fake images are real
            label.fill_(real_label)

            # Compute the loss using set criterion
            g_loss = e.criterion(output, label)

            # Define prediction as the rounded value from output
            g_pred = output.round()

            # Calcualte gradients for the generator
            g_loss.backward()
            d_g_z2 = output.mean().item()

            # Now update generator
            e.g_optimizer.step()

            # Update statistics
            g_batch_loss = g_loss.item()
            g_corrects = torch.sum(g_pred == 1).item()
            epoch_stats["g_train_loss"] += g_batch_loss
            epoch_stats["g_train_corrects"] += g_corrects

            # Print batch stats
            if batch_num % 50 == 0:
                logging.info(
                    "[{}/{}][{}/{}]\t[D loss: {:.4f}]\t[G loss: {:.4f}]\tD(x):{:.4f}\tD(G(z)):{:.4f}/{:.4f}".format(
                        epoch+1, 
                        e.params["num_epochs"],
                        batch_num + 1,
                        train_size + 1,
                        d_batch_loss,
                        g_batch_loss,
                        d_x, d_g_z1, d_g_z2))

            # Save batch losses plotting later
            e.training["d_batch_losses"].append(d_batch_loss)
            e.training["g_batch_losses"].append(g_batch_loss)


            # Check the progress of the generator by saving its output on fixed_noise
            if (batch_num % 500 == 0 ) or (epoch == 0 and batch_num % 50 == 0) or ((epoch == e.params["num_epochs"] - 1) and (batch_num == train_size - 1)):
                with torch.no_grad():
                    if flatten_input:
                        noise = fixed_noise.view(-1, e.params["nz"])
                    fake_img = e.generator(noise).detach().cpu()

                    
                if flatten_input: # if the image was flattened we need to deflatten it.
                    fake_img = fake_img.view(-1, e.params["nc"], e.params["im_size"], e.params["im_size"])
                e.training["image_list"].append(vutils.make_grid(fake_img, padding=2, normalize=True))

            # Increate iterations, number of batches processed
            iteration += 1 

        g_epoch_loss = epoch_stats["d_train_loss"]/train_size
        g_epoch_acc  = epoch_stats["d_train_corrects"]/(train_size*e.params["batch_size"]*2)
        d_epoch_loss = epoch_stats["g_train_loss"]/train_size
        d_epoch_acc  = epoch_stats["g_train_corrects"]/(train_size*e.params["batch_size"])
            
        # After finishing an epoch print the epoch stats
        logging.info("Epoch {}/{}: \t[D loss: {:.4f}, acc.:{:.4f}]\t[G loss: {:.4f}, acc.:{:.4f}]".format(
            epoch+1, 
            e.params["num_epochs"],
            d_epoch_loss,
            d_epoch_acc,
            g_epoch_loss,
            g_epoch_acc))

        # Save epoch stats for plotting later
        e.training["d_epoch_losses"].append(d_epoch_loss)
        e.training["d_epoch_accuracies"].append(d_epoch_acc)
        e.training["d_epoch_losses"].append(g_epoch_loss)
        e.training["d_epoch_accuracies"].append(g_epoch_acc)

        # In the first 3 epochs, the generator easily fools (or cheats) the discrimnator with random
        # noise and acheives quite low loss, let's disregard that when finding the lowest loss.
        if epoch > 2:
            if g_epoch_loss < e._best_loss:
                logging.info("Found lower loss, saving checkpoint")
                save_checkpoint(e, epoch, iteration)
            
            if training._early_stopping(e, g_epoch_loss):
                return

        
##############################################################
# Plotting results
##############################################################


def plot_all(e):
    plot_batch_losses(e, "training_batch_losses.png")
    plot_training_stats(e, "training_statistics.png")
    plot_training_progression(e, "training_progression.gif")
    plot_real_fake_images(e, "real_fake_images.png")
        

def plot_batch_losses(e, filename):
    plt.figure()
    plt.title("Generator and Discriminator Batch Losses During Training")
    plt.plot(e.training["g_batch_losses"], label="Generator")
    plt.plot(e.training["d_batch_losses"], label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(e.fname(filename), bbox_inches="tight", pad_inches=0.2)

 
def plot_training_stats(e, filename):
    plt.figure()
    plt.title("Generator and Discriminator Epoch Losses During Training")
    plt.plot(e.training["g_epoch_accuracies"], label="Generator Accuracy")
    plt.plot(e.training["d_epoch_accuracies"], label="Discriminator Accuracy")
    plt.plot(e.training["g_epoch_losses"], label="Generator Loss")
    plt.plot(e.training["d_epoch_losses"], label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/ Loss")
    plt.legend()
    plt.savefig(e.fname(filename), bbox_inches="tight", pad_inches=0.2)


def plot_training_progression(e, filename):
    fig = plt.figure()
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in e.training["image_list"]]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.title("Generator Progression")
    ani.save(e.fname(filename), writer="imagemagick", fps=10)
 

def plot_real_fake_images(e, filename):
    real_batch = next(iter(e.train_loader))

    # Plot the real images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0], padding=5, normalize=True), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(e.training["image_list"][-1],(1,2,0)))
    plt.savefig(e.fname(filename), bbox_inches="tight", pad_inches=0.2)
