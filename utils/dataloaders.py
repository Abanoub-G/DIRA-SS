import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import datasets, transforms

import numpy as np
import random

import matplotlib.pyplot as plt

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms and samples generation even when batchsize > dataset size.
    """
    def __init__(self, tensors, transform=None, batch_size=64):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

        self.data_len = self.tensors[0].size(0)
        self.batch_size = batch_size

    def __getitem__(self, index):

        index = index % self.data_len #

        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return max(self.data_len, self.batch_size)
        # return self.tensors[0].size(0)

def custom_train_transform(tensor):

    # Convert the tensor to a PIL image
    image = transforms.ToPILImage()(tensor)

    # Define random transforms: rotation, random horizontal flip, and random crop
    random_rotation = transforms.RandomRotation(degrees=(-45, 45))  # Rotate between -45 and 45 degrees
    random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)  # Randomly flip horizontally with 50% probability
    random_vertical_flip = transforms.RandomVerticalFlip(p=0.5)  # Randomly flip vertically with 50% probability
    random_crop = transforms.RandomCrop(size=(32, 32))  # Crop the image to a size of (64, 64)
    random_resized_crop = transforms.RandomResizedCrop(32, scale=(0.8,0.8))  # Crop the image to a size of (64, 64)

    # Normalize the tensor
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    normalize = transforms.Normalize(mean=mean, std=std)

    # Apply the transforms in a sequence
    transform = transforms.Compose([
        # random_rotation,
        random_horizontal_flip,
        random_vertical_flip,
        # random_crop,
        # random_resized_crop,
        transforms.ToTensor(),  # Converts the PIL image to a tensor
        normalize,  # Normalize the tensor
    ])

    # Apply the transforms to the image
    transformed_tensor = transform(image)

    # Convert the transformed image back to a tensor
    # transformed_tensor = transforms.ToTensor()(transformed_image)

    return transformed_tensor

def custom_test_transform(tensor):

    # Convert the tensor to a PIL image
    image = transforms.ToPILImage()(tensor)

    # Normalize the tensor
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    normalize = transforms.Normalize(mean=mean, std=std)

    # Apply the transforms in a sequence
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts the PIL image to a tensor
        normalize,  # Normalize the tensor
    ])

    # Apply the transforms to the image
    transformed_tensor = transform(image)

    # Convert the transformed image back to a tensor
    # transformed_tensor = transforms.ToTensor()(transformed_image)

    return transformed_tensor

def imshow(img, file_name, title=''):
    """Plot the image batch.
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(np.transpose( img.numpy(), (1, 2, 0)), cmap='gray')
    plt.savefig(file_name)

def cifar_c_dataloader(severity, noise_type="gaussian_noise", dataset_choice ="CIFAR10"):
    if dataset_choice == "CIFAR10":
        labels = torch.from_numpy(np.load("../datasets/CIFAR10/CIFAR-10-C/labels.npy"))
        test_images = torch.from_numpy(np.load("../datasets/CIFAR10/CIFAR-10-C/"+noise_type+".npy"))
    elif dataset_choice == "CIFAR100":
        labels = torch.from_numpy(np.load("../datasets/CIFAR100/CIFAR-100-C/labels.npy"))
        test_images = torch.from_numpy(np.load("../datasets/CIFAR100/CIFAR-100-C/"+noise_type+".npy"))
    
    
    labels = labels.to(torch.long)
    images_size = 32  # 32 x 32

    # Filter based on Severity
    labels = labels[10000*(severity-1):(10000*(severity))]
    images = test_images[10000*(severity-1):(10000*(severity))]

    # Rearranging tensor for corrupte data to match format of un-corrupt.
    images = images.reshape((images.size()[0], images_size, images_size,3))
    images = np.transpose(images,(0,3,1,2)) # Custom transpose needed for ouput of courrputed and non courrputed to match (found emprically)
    

    images = images.float()/255

    # test_set_c = torch.utils.data.TensorDataset(images,labels)
    # testloader_c = torch.utils.data.DataLoader(test_set_c, batch_size=64, shuffle=False)

    train_set_c = CustomTensorDataset(tensors=(images, labels), transform=custom_train_transform)
    trainloader_c = torch.utils.data.DataLoader(train_set_c, batch_size=64, shuffle=False)

    test_set_c = CustomTensorDataset(tensors=(images, labels), transform=custom_test_transform)
    testloader_c = torch.utils.data.DataLoader(test_set_c, batch_size=64, shuffle=False)

    image, label = train_set_c[0]
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    plt.imshow(image)
    plt.savefig("temp.png")
    # input("press enter to conti")

    return trainloader_c, testloader_c, images, labels

def imagenet_c_dataloader(severity, noise_type="gaussian_noise", tiny_imagenet=False):

    if tiny_imagenet == True:
        dataset_root = "../datasets/TinyImageNet/Tiny-ImageNet-C/"+noise_type+"/"+str(severity)+"/"
        train_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                  ])

        test_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                  ])
    else:
        dataset_root = "../datasets/ImageNet/ImageNet-C/"+noise_type+"/"+str(severity)+"/" #"../../datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

        test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

    train_set  = torchvision.datasets.ImageFolder(root=dataset_root, transform=train_transform)#ImageNetKaggle(dataset_root, "train", transform)
    test_set   = torchvision.datasets.ImageFolder(root=dataset_root, transform=test_transform)#ImageNetKaggle(dataset_root, "val", transform)



    trainloader_c = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    testloader_c = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)



    return trainloader_c, testloader_c, train_set, test_set

def mnist_c_dataloader(noise_type="gaussian_noise"):

   path_corrupt_dataset = '../datasets/MNIST/mnist_c/'

   train_images  = torch.from_numpy(np.load(path_corrupt_dataset+noise_type+"/train_images.npy"))
   train_labels  = torch.from_numpy(np.load(path_corrupt_dataset+noise_type+"/train_labels.npy"))
   test_images   = torch.from_numpy(np.load(path_corrupt_dataset+noise_type+"/test_images.npy"))
   test_labels   = torch.from_numpy(np.load(path_corrupt_dataset+noise_type+"/test_labels.npy"))

   # Rearranging tensor for corrupted data to match format of un-corrupt.
   train_images  = train_images.reshape((train_images.size()[0], 1, 28, 28)) 
   train_images  = train_images.float()/255
   test_images   = test_images.reshape((test_images.size()[0], 1, 28, 28)) 
   test_images   = test_images.float()/255
   # print(train_images.type())
   
   # == Load batches
   train_set_c   = torch.utils.data.TensorDataset(train_images,train_labels)
   trainloader_c = torch.utils.data.DataLoader(train_set_c, batch_size=64, shuffle=True)
   test_set_c    = torch.utils.data.TensorDataset(test_images,test_labels)
   testloader_c  = torch.utils.data.DataLoader(test_set_c, batch_size=64, shuffle=True)

   return testloader_c, trainloader_c

def pytorch_dataloader(dataset_name="", dataset_dir="", images_size=32, batch_size=64, retraining_imagenet = False):

    train_transform = transforms.Compose([
                transforms.Resize((images_size, images_size)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
              ])

    test_transform = transforms.Compose([
                transforms.Resize((images_size, images_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
              ])

    # Check which dataset to load.
    if dataset_name == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=train_transform) 
        test_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=test_transform)
        small_test_set = test_set

    elif dataset_name =="CIFAR100": 
        train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=train_transform) 
        test_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=test_transform)
        small_test_set = test_set

    elif dataset_name == "TinyImageNet":

        # train_transform = transforms.Compose([
        #                 transforms.ToTensor(),
        #                 ])

        # test_transform = transforms.Compose([
        #                 transforms.ToTensor(),
        #                 ])

        train_set = torchvision.datasets.ImageFolder(root=dataset_dir+"tiny-imagenet-200/train", transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(root=dataset_dir+"tiny-imagenet-200/val", transform=test_transform)
        small_test_set = test_set

    elif dataset_name =="ImageNet":
        dataset_root =  dataset_dir #"../../datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

        test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

        train_set  = torchvision.datasets.ImageFolder(root=dataset_root+"/train", transform=train_transform)#ImageNetKaggle(dataset_root, "train", transform)
        test_set   = torchvision.datasets.ImageFolder(root=dataset_root+"/val", transform=test_transform)#ImageNetKaggle(dataset_root, "val", transform)

        # Create a smaller subset of test images to speedup evaluation during retraining
        samples_indices_array = np.random.randint(0, 49999, size=1000)
        small_test_set = torch.utils.data.Subset(test_set, samples_indices_array) 

    else:
        print("ERROR: dataset name is not integrated into NETZIP yet.")


    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)

    small_test_loader = torch.utils.data.DataLoader(dataset=small_test_set, batch_size=64, shuffle=True)

    return train_loader, test_loader, small_test_loader


def samples_dataloader(N_T, noisy_images, noisy_labels):
    # Get length of images
    max_num_noisy_samples = len(noisy_labels)-1
    samples_indices_array = []
    for _ in range(N_T): 
        # Select a random number from the max number of images
        i = random.randint(0,max_num_noisy_samples)
        samples_indices_array.append(i)
    
    selected_noisy_images = np.take(noisy_images,samples_indices_array, axis=0)
    selected_noisy_labels = np.take(noisy_labels,samples_indices_array, axis=0)
        
    N_T_test_set_c = torch.utils.data.TensorDataset(selected_noisy_images,selected_noisy_labels)
    N_T_testloader_c = torch.utils.data.DataLoader(N_T_test_set_c, batch_size=64, shuffle=True)

    return N_T_testloader_c

def samples_dataloader_iterative(N_T, noisy_images, noisy_labels, samples_indices_array, N_T_step):
    # Get length of images
    max_num_noisy_samples = len(noisy_labels)-1

    for _ in range(N_T_step): 
        # Select a random number from the max number of images
        i = random.randint(0,max_num_noisy_samples)
        samples_indices_array.append(i)
    
    selected_noisy_images = np.take(noisy_images,samples_indices_array, axis=0)
    selected_noisy_labels = np.take(noisy_labels,samples_indices_array, axis=0)
        
    N_T_test_set_c = torch.utils.data.TensorDataset(selected_noisy_images,selected_noisy_labels)
    N_T_testloader_c = torch.utils.data.DataLoader(N_T_test_set_c, batch_size=64, shuffle=True)

    return N_T_testloader_c, samples_indices_array

def augmented_samples_dataloader_iterative(N_T, noisy_images, noisy_labels, samples_indices_array, N_T_step):
    # Get length of images
    max_num_noisy_samples = len(noisy_labels)-1

    for _ in range(N_T_step): 
        # Select a random number from the max number of images
        i = random.randint(0,max_num_noisy_samples)
        samples_indices_array.append(i)
    
    selected_noisy_images = np.take(noisy_images,samples_indices_array, axis=0)
    selected_noisy_labels = np.take(noisy_labels,samples_indices_array, axis=0)

    N_T_train_set_c = CustomTensorDataset(tensors=(selected_noisy_images, selected_noisy_labels), transform=custom_train_transform)
    N_T_trainloader_c = torch.utils.data.DataLoader(N_T_train_set_c, batch_size=64, shuffle=False)

    N_T_test_set_c = CustomTensorDataset(tensors=(selected_noisy_images, selected_noisy_labels), transform=custom_test_transform)
    N_T_testloader_c = torch.utils.data.DataLoader(N_T_test_set_c, batch_size=64, shuffle=False)


    # Display some images to visualise transforms
    show_pics_samples = True
    if show_pics_samples == True:
        for i, data in enumerate(N_T_trainloader_c):
            x, y = data  
            imshow(torchvision.utils.make_grid(x, 4), "train_transforms.png" , title='train_Transforms')
            break
        for i, data in enumerate(N_T_testloader_c):
            x, y = data  
            imshow(torchvision.utils.make_grid(x, 4), "test_transforms.png", title='test_Transforms')
            break

    return N_T_trainloader_c, N_T_testloader_c, samples_indices_array

def augmented_samples_dataloader_iterative_imagenet(N_T, train_set, test_set, samples_indices_array, N_T_step):
    # Get length of images
    max_num_noisy_samples = len(test_set)-1

    for _ in range(N_T_step): 
        # Select a random number from the max number of images
        i = random.randint(0,max_num_noisy_samples)
        samples_indices_array.append(i)

    selected_train_subset = torch.utils.data.Subset(train_set, samples_indices_array)
    selected_test_subset = torch.utils.data.Subset(test_set, samples_indices_array)
    
    # selected_noisy_images = np.take(noisy_images,samples_indices_array, axis=0)
    # selected_noisy_labels = np.take(noisy_labels,samples_indices_array, axis=0)

    N_T_trainloader_c = torch.utils.data.DataLoader(selected_train_subset, batch_size=64, shuffle=False)
    N_T_testloader_c = torch.utils.data.DataLoader(selected_test_subset, batch_size=64, shuffle=False)


    # Display some images to visualise transforms
    show_pics_samples = True
    if show_pics_samples == True:
        for i, data in enumerate(N_T_trainloader_c):
            x, y = data  
            imshow(torchvision.utils.make_grid(x, 4), "train_transforms.png" , title='train_Transforms')
            break
        for i, data in enumerate(N_T_testloader_c):
            x, y = data  
            imshow(torchvision.utils.make_grid(x, 4), "test_transforms.png", title='test_Transforms')
            break

    return N_T_trainloader_c, N_T_testloader_c, samples_indices_array



# Custom dataset class for CIFAR-10 with rotation labels
class CIFAR10WithRotation(Dataset):
    def __init__(self, root, train=True, download=True, transform=None, do_rotations=True):
        self.cifar10 = torchvision.datasets.CIFAR10(root, train=train, download=download, transform=transform)
        self.do_rotations = do_rotations

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        
        # Randomly generate a rotation label (0, 1, 2, or 3)
        if self.do_rotations:
            rotation_label = np.random.randint(4)  # Randomly select 0, 1, 2, or 3
        else:
            rotation_label = 0
        
        # Apply rotation to the image based on the label
        if rotation_label == 1:
            image = torch.rot90(image,1, [1, 2])
        elif rotation_label == 2:
            image = torch.rot90(image,2, [1, 2])
        elif rotation_label == 3:
            image = torch.rot90(image,3, [1, 2])
        
        return image, rotation_label


def pytorch_rotation_dataloader(dataset_name="", dataset_dir="", images_size=32, batch_size=64):

    train_transform = transforms.Compose([
                transforms.Resize((images_size, images_size)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
              ])

    test_transform = transforms.Compose([
                transforms.Resize((images_size, images_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
              ])

    # Check which dataset to load.
    if dataset_name == "CIFAR10":
        train_set = CIFAR10WithRotation(root=dataset_dir, train=True, download=True, transform=train_transform) 
        test_set  = CIFAR10WithRotation(root=dataset_dir, train=False, download=True, transform=test_transform)
    
    elif dataset_name =="CIFAR100": 
        train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=train_transform) 
        test_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "TinyImageNet":

        # train_transform = transforms.Compose([
        #                 transforms.ToTensor(),
        #                 ])

        # test_transform = transforms.Compose([
        #                 transforms.ToTensor(),
        #                 ])

        train_set = torchvision.datasets.ImageFolder(root=dataset_dir+"tiny-imagenet-200/train", transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(root=dataset_dir+"tiny-imagenet-200/val", transform=test_transform)
    
    elif dataset_name =="ImageNet":
        dataset_root =  dataset_dir #"../../datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

        test_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ])

        train_set  = torchvision.datasets.ImageFolder(root=dataset_root+"/train", transform=train_transform)#ImageNetKaggle(dataset_root, "train", transform)
        test_set   = torchvision.datasets.ImageFolder(root=dataset_root+"/val", transform=test_transform)#ImageNetKaggle(dataset_root, "val", transform)


    else:
        print("ERROR: dataset name is not integrated into NETZIP yet.")


    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)

    return train_loader, test_loader

# def pytorch_dataloader_with_rotation(dataset_name="", dataset_dir="", images_size=32, batch_size=64, do_rotations=True):

#     train_transform = transforms.Compose([
#                 transforms.Resize((images_size, images_size)),
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#               ])

#     test_transform = transforms.Compose([
#                 transforms.Resize((images_size, images_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#               ])

#     # Check which dataset to load.
#     if dataset_name == "CIFAR10":
#         train_set = CIFAR10WithRotation(root=dataset_dir, train=True, download=True, transform=train_transform, do_rotations = do_rotations) 
#         test_set  = CIFAR10WithRotation(root=dataset_dir, train=False, download=True, transform=test_transform, do_rotations = do_rotations)
    
#     elif dataset_name =="CIFAR100": 
#         train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=train_transform) 
#         test_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=test_transform)

#     elif dataset_name == "TinyImageNet":

#         # train_transform = transforms.Compose([
#         #                 transforms.ToTensor(),
#         #                 ])

#         # test_transform = transforms.Compose([
#         #                 transforms.ToTensor(),
#         #                 ])

#         train_set = torchvision.datasets.ImageFolder(root=dataset_dir+"tiny-imagenet-200/train", transform=train_transform)
#         test_set = torchvision.datasets.ImageFolder(root=dataset_dir+"tiny-imagenet-200/val", transform=test_transform)
    
#     elif dataset_name =="ImageNet":
#         dataset_root =  dataset_dir #"../../datasets/ImageNet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
#         mean = (0.485, 0.456, 0.406)
#         std = (0.229, 0.224, 0.225)

#         train_transform = transforms.Compose([
#                         transforms.Resize(256),
#                         transforms.CenterCrop(224),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean, std),
#                     ])

#         test_transform = transforms.Compose([
#                         transforms.Resize(256),
#                         transforms.CenterCrop(224),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean, std),
#                     ])

#         train_set  = torchvision.datasets.ImageFolder(root=dataset_root+"/train", transform=train_transform)#ImageNetKaggle(dataset_root, "train", transform)
#         test_set   = torchvision.datasets.ImageFolder(root=dataset_root+"/val", transform=test_transform)#ImageNetKaggle(dataset_root, "val", transform)


#     else:
#         print("ERROR: dataset name is not integrated into NETZIP yet.")


#     train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)

#     test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)

#     return train_loader, test_loader
