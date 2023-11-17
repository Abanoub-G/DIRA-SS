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

path = "./CIFAR100/"
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

train_set = datasets.CIFAR100(path, download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.CIFAR100(path, download=True, train=False, transform=transform)
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
url = 'https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1'

if not os.path.isfile(path+"CIFAR-100-C.tar"):
   wget.download(url, path)

if not os.path.isdir(path+"CIFAR-100-C"):
   my_tar = tarfile.open(path+"CIFAR-100-C.tar", "r:")
   my_tar.extractall(path)
   my_tar.close()
