import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np


import torch
from torchvision import datasets, transforms

import wget

import zipfile

path = "./MNIST/"
if not os.path.isdir(path):
   os.makedirs(path)

# ===========
# Dataset MNIST
# ===========
transform = transforms.Compose([
          transforms.Resize((28, 28)),
          transforms.ToTensor()
          ])

train_set = datasets.MNIST(path, download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.MNIST(path, download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
# print(train_set.size())
# print(trainloader.size())



# ===========
# Dataset MNIST-C
# ===========
# == Download Files 
url = 'https://zenodo.org/record/3239543/files/mnist_c.zip?download=1'

if not os.path.isfile(path+"mnist_c.zip"):
   wget.download(url, path)

if not os.path.isdir(path+"mnist_c"):
   with zipfile.ZipFile(path+"mnist_c.zip", 'r') as zip_ref:
      zip_ref.extractall(path)

# # == Load Files for a given noise
# train_images = torch.from_numpy(np.load(path+"mnist_c/fog/train_images.npy"))
# train_labels = torch.from_numpy(np.load(path+"mnist_c/fog/train_labels.npy"))
# test_images  = torch.from_numpy(np.load(path+"mnist_c/fog/test_images.npy"))
# test_labels  = torch.from_numpy(np.load(path+"mnist_c/fog/test_labels.npy"))

# # Rearranging tensor for corrupte data to match format of un-corrupt.
# train_images = train_images.reshape((train_images.size()[0], 1, 28, 28)) 
# train_images = train_images.float()
# test_images  = test_images.reshape((test_images.size()[0], 1, 28, 28)) 
# test_images  = test_images.float()

# # == Load batches
# train_set_c = torch.utils.data.TensorDataset(train_images,train_labels)
# trainloader_c = torch.utils.data.DataLoader(train_set_c, batch_size=64, shuffle=True)
# test_set_c = torch.utils.data.TensorDataset(test_images,test_labels)
# testloader_c = torch.utils.data.DataLoader(test_set_c, batch_size=64, shuffle=True)


# tensor_image1, index_label = train_set[40]
# print("mnist type = ",tensor_image1.type())
# # print(tensor_image1.size())
# new_img1 = np.transpose(tensor_image1, (1,2,0)).reshape((28,28,tensor_image1.size()[0]))
# plt.clf()
# plt.imshow(new_img1)
# plt.savefig("mnist_trial.png")

# tensor_image1, index_label = train_set_c[40]
# print("mnist_c type = ",tensor_image1.type())
# # print(tensor_image1.size())
# new_img1 = np.transpose(tensor_image1, (1,2,0)).reshape((28,28,tensor_image1.size()[0]))
# plt.clf()
# plt.imshow(new_img1)
# plt.savefig("mnist_trial_c.png")

