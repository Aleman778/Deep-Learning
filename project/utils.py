import os, shutil
import torch
import torchvision.utils as vutils
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
    print("Training on:", device_name)
    return device


def image_transforms(im_size):
    """Returns a simple image transform resizes the image to the given size
    and crops the image, converts it to tensor and finally normalize it around 0.5"""
    return transforms.Compose([
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5,  0.5), (0.5, 0.5, 0.5))
    ])


def plot_dataset(path, dataloder, num_samples=32, show_labels=True, figsize=(12, 8)):
    inputs, classes = next(iter(datalaoder));
    inputs = vutils.make_grid(inputs, padding=2, normalize=True)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(np.transpose(inputs, (1, 2, 0)))
    title = ""
    for i, x in enumerate(classes):
        title += class_names[x] + ", " 
        if (i % 8) == 7:
            title += "\n"
    plt.title(title)
    plt.savefig(path)



class Experiment:
    """Experiment class defines basic contents of an experiment.
    This is defined by the folder where results are stored and it
    can also be used to store training and testing data."""
    def __init__(self, folder):
        self.folder = folder
        self.training = {}
        self.testing = {}
    

    def fname(self, name):
        return os.path.join(self.folder, name)
    

        
def create_experiment(base_path, experiment_callback):
    """Creates a new experiment using a simple CLI.
    The user is promted to specifiy the name of the experiment."""
    while True:
        experiment_name = input("Experiment name: ")
        if experiment_name == "":
            print("No name was given, exiting.")
            return
        
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
    experiment = Experiment(folder)
    experiment_callback(experiment)
    


