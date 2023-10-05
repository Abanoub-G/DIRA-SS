import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch.optim.lr_scheduler as lr_scheduler

import torch
from torch import nn, optim
from torchvision import datasets, transforms

from utils.common import set_random_seeds, set_cuda


from utils.model import model_selection, train_model, save_model
from utils.dataloaders import pytorch_dataloader
from metrics.accuracy.topAccuracy import top1Accuracy


# =====================================================
# == Declarations
# =====================================================
SEED_NUMBER              = 0
USE_CUDA                 = True

DATASET_DIR              = '../datasets/CIFAR10/'
DATASET_NAME             = "CIFAR10" # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES              = 1000 # Number of classes in dataset

MODEL_CHOICE             = "resnet" # Option:"resnet" 
MODEL_VARIANT            = "resnet18" # Common Options: "resnet18" "resnet26" For more options explore files in models to find the different options.

MODEL_DIR                = "../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

SAVED_MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+".pt"
SAVED_MODEL_FILEPATH     = os.path.join(MODEL_DIR, SAVED_MODEL_FILENAME)

TRAINED_MODEL_FILENAME   = MODEL_VARIANT +"_"+DATASET_NAME+".pt"

NUM_EPOCHS               = 3
LEARNING_RATE            = 1e-2


# Fix seeds to allow for repeatable results 
set_random_seeds(SEED_NUMBER)

# Setup device used for training either gpu or cpu
device = set_cuda(USE_CUDA)


def main():
    # Fix seeds to allow for repeatable results 
    set_random_seeds(SEED_NUMBER)

    # Setup device used for training either gpu or cpu
    device = set_cuda(USE_CUDA)

    # Setup dataset
    trainloader, testloader = pytorch_dataloader(dataset_name=DATASET_NAME, dataset_dir=DATASET_DIR, images_size=32, batch_size=64)
    print("Progress: Dataset Loaded.")

    # Setup model
    model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=SAVED_MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device)
    print("Progress: Model has been setup.")

    # Train model
    model = train_model(model=model, train_loader=trainloader, test_loader=testloader, device=device, learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)
    print("Progress: Model training done.")

    # Save model.
    save_model(model=model, model_dir=MODEL_DIR, model_filename=TRAINED_MODEL_FILENAME)
    print("Progress: Model Saved.")

    # Evaluate model
    _,eval_accuracy     = top1Accuracy(model=model, test_loader=testloader, device=device, criterion=None)
   
    print("FP32 evaluation accuracy: {:.3f}".format(eval_accuracy))
    
if __name__ == "__main__":

    main()
