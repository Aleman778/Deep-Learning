"""Training module defines some reusable functions in training neural nets
"""

import os
import logging
import torch


def _save_checkpoint(e, data, filename="checkpoint.pth"):
    """Saves a checkpoint of the experiment from dictionary of 
    custom items and stores with the provided filename.
    NOTE: this function should not be directly called by the experiment.
    Use the specific training modules save_checkpoint instead.
    """
    data["training"] = e.training
    data["testing"] = e.testing
    data["params"] = e.params
    torch.save(data, e.fname(filename))
    logging.info("Checkpoint saved to `" + filename + "`")
    

def _load_checkpoint(e, path):
    """Loads a specfic experiment checkpoint, returns the experiment and
    a checkpoint dictionary previously stored by the _save_checkpoint function.
    NOTE: this function should not be directly called by the experiment
    use the specific training modules load_checkpoint instead."""

    if not os.path.isfile(path):
        logging.error("Could not find checkpoint file `" + path + "`, exiting.")
        exit(0)
        return None
        
    checkpoint = torch.load(path)
    e.training = checkpoint["training"]
    e.testing = checkpoint["testing"]
    e.params = checkpoint["params"]
    return checkpoint


def _early_stopping(e, current_loss):
    """Returns true if the training should stop, due to decreasing loss."""
    if current_loss < e._best_loss:
        e._best_loss = current_loss
        e._best_loss_counter = 0
        return False

    if not "patience" in e.params: return True
    
    e._best_loss_counter += 1
    if e._best_loss_counter >= e.params["patience"]:
        logging.info("Early stopping, loss has not decreased for " + str(e.params["patience"]) + " epochs")
        return True
    return False

    
    
