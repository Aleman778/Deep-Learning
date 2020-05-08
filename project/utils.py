import os, shutil
import time
import random
import numpy as np
import logging
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision import transforms


"""The utils module defines helper functions and can easily deal
with reproducable experiments. Makes it effortless to create new experiments.
"""

def cuda_device_if_available():
    """Returns the CUDA device if it is available, 
    if no then the CPU device is returned instead.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logging.info("Training on: " + device_name)
    return device


def image_transform(im_size, nc=3):
    """Returns a simple image transform resizes the image to the given size
    and crops the image, converts it to tensor and finally normalize it around 0.5"""
    return transforms.Compose([
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
    ])



def plot_data_subset(path, dataset, indices=range(32), show_labels=True, figsize=(12, 8)):
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=len(subset))
    inputs, classes = next(iter(dataloader)) 
    inputs = vutils.make_grid(inputs, padding=2, normalize=True)

    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(np.transpose(inputs, (1, 2, 0)))
    if show_labels:
        title = ""
        for i, x in enumerate(classes):
            title += dataset.classes[x] + ", " 
            if (i % 8) == 7:
                title += "\n"
        plt.title(title)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.2)



class Experiment:
    """Experiment class defines all the contents of an experiment.
    It is a container for all possible parameters and data recorded."""
    def __init__(self, name, folder, loglevel, seed):
        self.name = name
        self.folder = folder
        self.train_loader = None
        self.eval_loader = None
        self.test_loader = None
        self.num_epochs = 0
        self.params = {}
        self.training = {}
        self.testing = {}
        self._best_loss = 1000000 # used for early stopping, and storing best model
        self._best_loss_count = 0 # used for early stopping
        logging.basicConfig(filename=os.path.join(folder, "log.txt"),
                            filemode='a',
                            format='%(asctime)s %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=loglevel)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Initialzed experiment: " + str(name))
        self.device = cuda_device_if_available()
        
        # Setting seed for RNGs (set to constant for reproducability).
        if seed == None: seed = time.time()
        logging.info("Initialize seed: " + str(seed))
        random.seed(seed)
        torch.manual_seed(seed)
        

    

    def fname(self, name):
        """Joins the experiment directory with the filename, returns the file path."""
        return os.path.join(self.folder, name)


    def setup_dataloader(self, datasets):
        if datasets[0] != None:
            self.train_loader = torch.utils.data.DataLoader(datasets[0],
                                                            batch_size=self.params["batch_size"],
                                                            shuffle=self.params["shuffle"],
                                                            num_workers=self.params["num_workers"])

            
        if datasets[1] != None:
            self.eval_loader = torch.utils.data.DataLoader(datasets[1],
                                                           batch_size=self.params["batch_size"],
                                                           shuffle=self.params["shuffle"],
                                                           num_workers=self.params["num_workers"])

        
        if datasets[2] != None:
            self.test_loader = torch.utils.data.DataLoader(datasets[2],
                                                           batch_size=self.params["batch_size"],
                                                           shuffle=self.params["shuffle"],
                                                           num_workers=self.params["num_workers"])
    
    def input_int(self, message, default=None):
        """Input for integers, optionally provide default value."""
        message = "Enter the " + message
        if default != None: message += " [default: " + str(default) + "]"
        while True:
            value = input(message + ": ")
            if value == "" and default != None:
                return default
            
            try:
                value = int(value)
            except:
                print("Invalid number entered, please try again.")
                continue

            return value


def create_experiment(base_path, loglevel=logging.INFO, seed=None):
    """Creates a new experiment using a simple CLI.
    The user is promted to specifiy the name of the experiment."""
    while True:
        experiment_name = input("Experiment name: ")
        if experiment_name == "":
            print("No name was given, exiting.")
            exit(0)
        
        folder = os.path.join(base_path, experiment_name)
        if os.path.isdir(folder):
            
            overwrite = input("There already exists an experiment with this name, " +
                              "do you want to overwrite it? [y,N] ")
            if overwrite == "y":
                if shutil.rmtree.avoids_symlink_attacks:
                    print("Cannot delete the old files, please do it manually and retry.")
                    continue

                shutil.rmtree(folder)
                print("Removed old experiment results.")
                break
            else:
                print("Please pick another name.")
        else:
            break
    
    os.makedirs(folder)
    return Experiment(experiment_name, folder, loglevel, seed)
    


