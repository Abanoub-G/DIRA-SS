import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np


import torch
from torchvision import datasets, transforms

import wget

import zipfile
import tarfile

path = "./CIFAR10/"
if not os.path.isdir(path):
   os.makedirs(path)
images_size = 32 # 32 x 32
# ===========
# Dataset CIFAR10
# ===========
transform = transforms.Compose([
          transforms.Resize((images_size, images_size)),
          transforms.ToTensor()
          ])

train_set = datasets.CIFAR10(path, download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.CIFAR10(path, download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)


# tensor_image1, index_label = test_set[0]
# new_img1 = np.transpose(tensor_image1, (1,2,0)).reshape((images_size,images_size,tensor_image1.size()[0]))
# plt.clf()
# plt.imshow(new_img1)
# plt.savefig("cifar_trial.png")

# ===========
# Dataset CIFAR10-C
# ===========
# == Download Files 
url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1'

if not os.path.isfile(path+"CIFAR-10-C.tar"):
   wget.download(url, path)

if not os.path.isdir(path+"CIFAR-10-C"):
   my_tar = tarfile.open(path+"CIFAR-10-C.tar", "r:")
   my_tar.extractall(path)
   my_tar.close()

# # == Load Files for a given noise and severity level
# # severity_level = 2 # From 1 to 5
# labels = torch.from_numpy(np.load(path+"CIFAR-10-C/labels.npy"))
# test_images = torch.from_numpy(np.load(path+"CIFAR-10-C/gaussian_noise.npy"))

# for severity_level in range(1,6):
#    # Select Severity
#    current_labels = labels[10000*(severity_level-1):(10000*(severity_level))]
#    corrupt_images = test_images[10000*(severity_level-1):(10000*(severity_level))]
#    # print("current test images size = ",corrupt_images.size())
   
#    # Rearranging tensor for corrupte data to match format of un-corrupt.
#    corrupt_images = corrupt_images.reshape((corrupt_images.size()[0], 3, images_size, images_size))
#    corrupt_images = np.transpose(corrupt_images,(0,3,1,2)) # Custom transpose needed for ouput of courrputed and non courrputed to match (found emprically)
#    corrupt_images = corrupt_images.float()/255


#    # == Load batches
#    test_set_c = torch.utils.data.TensorDataset(corrupt_images,current_labels)
#    testloader_c = torch.utils.data.DataLoader(test_set_c, batch_size=64, shuffle=True)

#    tensor_image1, index_label = test_set_c[0]
#    new_img1 = np.transpose(tensor_image1, (1,2,0)).reshape((images_size,images_size,3))
#    plt.clf()
#    plt.imshow(new_img1)
#    plt.savefig("cifar_trial_"+str(severity_level)+".png")

# def cifar10c_dataloader(severity, noise_type="gaussian_noise"):
#    labels = torch.from_numpy(np.load("./CIFAR10/CIFAR-10-C/labels.npy"))
#    test_images = torch.from_numpy(np.load("./CIFAR10/CIFAR-10-C/"+noise_type+".npy"))
#    images_size = 32  # 32 x 32

#    # Filter based on Severity
#    labels = labels[10000*(severity-1):(10000*(severity))]
#    images = test_images[10000*(severity-1):(10000*(severity))]
    
#    # Rearranging tensor for corrupte data to match format of un-corrupt.
#    images = images.reshape((images.size()[0], 3, images_size, images_size))
#    images = np.transpose(images,(0,3,1,2)) # Custom transpose needed for ouput of courrputed and non courrputed to match (found emprically)
#    images = images.float()/255

#    test_set_c = torch.utils.data.TensorDataset(images,labels)
#    testloader_c = torch.utils.data.DataLoader(test_set_c, batch_size=64, shuffle=True)

#    tensor_image1, index_label = test_set_c[0]
#    new_img1 = np.transpose(tensor_image1, (1,2,0)).reshape((images_size,images_size,3))
#    plt.clf()
#    plt.imshow(new_img1)
#    plt.savefig("cifar_c_trial.png")


#    return testloader_c

# cifar10c_dataloader(5, "impulse_noise")



