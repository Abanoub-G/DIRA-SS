import sys
sys.path.append('../')

import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.common import set_random_seeds, set_cuda
from utils.model import model_selection


SEED_NUMBER              = 0
USE_CUDA                 = True

DATASET_DIR              = '../datasets/CIFAR10/'
DATASET_NAME             = "CIFAR10" # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES              = 1000 # Number of classes in dataset

MODEL_CHOICE             = "resnet" # Option:"resnet" 
MODEL_VARIANT            = "resnet18" # Common Options: "resnet18" "resnet26" For more options explore files in models to find the different options.

MODEL_DIR                = "../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

# SAVED_MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+".pt"
# SAVED_MODEL_FILEPATH     = os.path.join(MODEL_DIR, SAVED_MODEL_FILENAME)

# Fix seeds to allow for repeatable results 
set_random_seeds(SEED_NUMBER)

# Setup device used for training either gpu or cpu
device = set_cuda(USE_CUDA)

# List of models
# initial_model_file_name = "resnet18_CIFAR10.pt"
models_list = []
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pt"):
        if file == "resnet18_CIFAR10_multi.pt":
            pass
        else:
            models_list.append(file)

# Initialize a list to store mean weights for each layer in each model
mean_weights_per_model = []

# Loop over models
for model_name in models_list:
    model_filepath = os.path.join(MODEL_DIR, model_name)

    print(model_name)

    # Load retrained model
    model = model_selection(model_selection_flag=2, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=model_filepath, num_classes=NUM_CLASSES, device=device)

    model.eval()  # Set the model to evaluation mode
    
    # Initialize a list to store mean weights for each layer
    mean_weights_list = []
    
    # Iterate through all layers in the model
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            # Get the weight tensor of the convolutional layer
            weights = layer.weight.data.cpu().numpy()
            
            # Calculate the mean absolute value for each filter's weights
            mean_weights = np.mean(np.abs(weights), axis=(1, 2, 3))
            
            # Store the mean weights along with the layer name
            mean_weights_list.append((name, mean_weights))
    
    # Append the mean weights for this model to the list
    mean_weights_per_model.append(mean_weights_list)

# === Calculate the variance of mean weights across different models for each layer
variance_per_layer = []
for i in range(len(mean_weights_per_model[0])):
    layer_name = mean_weights_per_model[0][i][0]
    mean_weights = [model_weights[i][1] for model_weights in mean_weights_per_model]
    variance = np.var(mean_weights, axis=0)
    variance_per_layer.append((layer_name, variance))


# Create a colormap with N distinct colors
N=len(variance_per_layer)
cmap = plt.cm.get_cmap('tab20c', N) #tab20
# Generate N equally spaced values from 0 to 1
values = np.linspace(0, 1, N)
# Get RGBA colors from the colormap
colors = [cmap(value) for value in values]

# Plot the variance for each layer
plt.figure(figsize=(12, 8))
for i, (name, variance) in enumerate(variance_per_layer):
    plt.subplot(4, 5, i + 1)
    plt.bar(range(len(variance)), variance, color=colors[i])
    plt.title(i, fontsize = 8)#name, fontsize = 8)
    plt.xlabel('Filter Index')
    plt.ylabel('Variance') # Variance of Mean weight value
    # plt.ylim(0,1e-7)
    plt.yscale("log") 
    plt.ylim(1e-13,1e-6)
    # plt.minorticks_on()
    # plt.grid(axis = 'y', which="both")

plt.tight_layout()
plt.savefig("variance.pdf")
plt.savefig("variance.png")

# === Create a 3D plot for the variances of each layer
plt.clf()
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
tick_labels = []
i_loc = []
for i, (name, variance) in enumerate(variance_per_layer):
    x = np.arange(len(variance))
    y = x*0+i
    z = variance

    # plt.fill_between(x, z, color='skyblue')
    ax.plot3D(x,y,z, color=colors[i])

    i_loc.append(i)
    tick_labels.append(name)

# plt.yticks(i_loc, tick_labels)

# plt.ylim(0,20)
plt.yticks(np.arange(0, 20, step=1))
ax.set_xlabel('Filter Index')
ax.set_ylabel('Layer Index')
ax.set_zlabel('Variance')
ax.set_title('Parameter Variance Across Layers')
plt.savefig("variance3D.pdf")
plt.savefig("variance3D.png")


# # === Create a 3D scatter plot for the variances
# plt.clf()
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# for i, (name, variance) in enumerate(variance_per_layer):
#     x = np.arange(len(variance))
#     y = np.full(len(variance), i)
#     z = variance
#     ax.scatter(x, y, z, label=name)

# ax.set_xlabel('Filter Index')
# ax.set_ylabel('Layer Index')
# ax.set_zlabel('Variance')
# ax.set_title('Parameter Variance Across Layers')
# ax.legend()
# plt.savefig("variance_scatter3D.pdf")
