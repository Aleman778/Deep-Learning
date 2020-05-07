import logging
import torch
import torchvision
import torchvision.utils as vutils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from torch import nn

         
def train_model(e):
    # Labels is defined as follows
    fake_label = 0
    real_label = 1

    # Basic dataset information
    train_size = len(e.train_loader)

    # Store some images during training, to see generator progression.
    img_list = []
    fixed_noise = torch.randn(64, e.params["nz"], 1, 1, device=e.device)

    # Keep track of some running statistics to view after training
    g_batch_losses = []
    d_batch_losses = []
    g_train_losses = []
    d_train_losses = []
    g_train_accuracies = []
    d_train_accuracies = []


    ##############################################################
    # Training loop
    ##############################################################

    for epoch in range(e.params["num_epochs"]):
        epoch_stats = {
            "discriminator_train_loss": 0.0,
            "discriminator_train_corrects": 0.0,
            "generator_train_loss": 0.0,
            "generator_train_corrects": 0.0}

        for batch_num, (real_images, _) in enumerate(e.train_loader):

            ##############################################################
            # Training Discriminator with REAL (dataset) images
            ##############################################################


            # Reset the discriminator gradients
            e.discriminator.zero_grad()

            # Setup the input data and label
            real_images = real_images.to(e.device);
            b_size = real_images.size(0)
            label = torch.full((b_size,), real_label, device=e.device);

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

            # Generate fake image batch using our generator
            fake_images = e.generator(noise)
            label.fill_(fake_label);

            # Make predictions for the batch of fake images using discriminator
            output = e.discriminator(fake_images).view(-1)

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
            d_corrects = torch.sum(d_pred_fake == 0).item()
            epoch_stats["discriminator_train_loss"] += d_batch_loss
            epoch_stats["discriminator_train_corrects"] += d_corrects


            ##############################################################
            # Training Generator
            ##############################################################


            # Reset the discriminator gradients
            e.generator.zero_grad()

            # Generate batch of new latent vectors
            noise = torch.randn(b_size, e.params["nz"], 1, 1, device=e.device)

            # Generate new fake image batch using our generator
            fake_images = e.generator(noise)

            # Since we just updated our discriminator we now want to perform
            # another forward pass of the batch of fake images trough our discriminator
            output = e.discriminator(fake_images).view(-1)

            # When training the generator we want the discriminator to think our fake images are real
            label.fill_(real_label);

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
            epoch_stats["generator_train_loss"] += g_batch_loss
            epoch_stats["generator_train_corrects"] += g_corrects

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
            d_batch_losses.append(d_batch_loss)
            g_batch_losses.append(g_batch_loss)


            # Check the progress of the generator by saving its output on fixed_noise
            if (batch_num % 500 == 0) or ((epoch == e.params["num_epochs"] - 1) and (batch_num == train_size - 1)):
                with torch.no_grad():
                    fake_img = e.generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_img, padding=2, normalize=True))

        # After finishing an epoch print the epoch stats
        logging.info("Epoch {}/{}: \t[D loss: {:.4f}, acc.:{:.4f}]\t[G loss: {:.4f}, acc.:{:.4f}]".format(
            epoch+1, 
            e.params["num_epochs"],
            batch_num + 1,
            train_size + 1,
            epoch_stats["discriminator_train_loss"]/train_size,
            epoch_stats["discriminator_train_corrects"]/train_size,
            epoch_stats["generator_train_loss"]/train_size,
            epoch_stats["generator_train_corrects"]/train_size))

        # Save epoch stats for plotting later
        d_train_losses.append(epoch_stats["discriminator_train_loss"]/train_size)
        d_train_accuracies.append(epoch_stats["discriminator_train_corrects"]/train_size)
        g_train_losses.append(epoch_stats["generator_train_loss"]/train_size)
        g_train_accuracies.append(epoch_stats["generator_train_corrects"]/train_size)

        e.training["discriminator_batch_losses"]     = d_batch_losses
        e.training["discriminator_train_losses"]     = d_train_losses
        e.training["discriminator_train_accuracies"] = d_train_losses
        e.training["generator_batch_losses"]         = g_batch_losses
        e.training["generator_train_losses"]         = g_train_losses
        e.training["generator_train_accuracies"]     = g_train_losses

def plot_batch_losses(e, filename):
    plt.figure(3)
    plt.title("Generator and Discriminator Batch Losses During Training")
    plt.plot(e.training["generator_batch_losses"], label="Generator")
    plt.plot(e.training["discriminator_batch_losses"], label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(e.fname(filename), bbox_inches='tight', pad_inches=0)


def plot_training_stats(e, filename):
    plt.figure(4)
    plt.title("Generator and Discriminator Epoch Losses During Training")
    plt.plot(e.training["generator_train_accuracies"], label="Generator Accuracy")
    plt.plot(e.training["discriminator_train_accuracies"], label="Discriminator Accuracy")
    plt.plot(e.training["generator_train_losses"], label="Generator Loss")
    plt.plot(e.training["discriminator_train_losses"], label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/ Loss")
    plt.legend()
    plt.savefig(e.fname(filename), bbox_inches='tight', pad_inches=0)
