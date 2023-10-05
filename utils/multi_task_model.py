import os

import importlib

import torch
import torch.nn as nn
import torch.optim as optim

from metrics.accuracy.topAccuracy import top1Accuracy, top1Accuracy_rotation

import matplotlib.pyplot as plt

# Define a multi-task ResNet architecture
# class MultiTaskResNet(nn.Module):
#     def __init__(self, model, num_classes, num_rotation_classes=4, classification_layers=512, rotation_layers=512):
#         super(MultiTaskResNet, self).__init__()

#         self.resnet = model

#         self.n_features = self.resnet.fc.in_features

#         self.resnet.fc = nn.Identity()

#         self.resnet.classification_head = nn.Sequential(
#                                                         nn.Linear(self.n_features,self.n_features),
#                                                         nn.ReLU(),
#                                                         nn.Linear(self.n_features, num_classes)
#                                                         )

#         self.resnet.rotation_head = nn.Sequential(
#                                                         nn.Linear(self.n_features,self.n_features),
#                                                         nn.ReLU(),
#                                                         nn.Linear(self.n_features, num_rotation_classes)
#                                                         )

#     def forward(self, x):
#         features = self.resnet(x)
#         classification_output = self.resnet.classification_head(features)
#         rotation_output = self.resnet.rotation_head(features)
#         return classification_output, rotation_output

# Define a multi-task ResNet architecture
class MultiTaskResNet(nn.Module):
    def __init__(self, model, num_classes, num_rotation_classes=4, classification_layers=512, rotation_layers=512):
        super(MultiTaskResNet, self).__init__()

        self.resnet = model

        self.n_features = self.resnet.fc.in_features

        
        self.classification_head = self.resnet.fc

        self.rotation_head = nn.Sequential(
                                                        nn.Linear(self.n_features,self.n_features),
                                                        nn.ReLU(),
                                                        nn.Linear(self.n_features, num_rotation_classes)
                                                        )

        self.num_classificaiton_classes = num_classes
        self.num_rotation_classes = num_rotation_classes

        # Initially, use the classification head
        self.use_classification_head()

    def forward(self, x):
        # Delegate the forward pass to the currently active head
        return self.resnet(x)

    def use_classification_head(self):
        # Switch to the original classification head
        self.resnet.fc = self.classification_head
        # self.active_head = self.classification_head
        # self.num_classes = self.num_classificaiton_classes

    def use_rotation_head(self):
        # Switch to the custom second head
        self.resnet.fc = self.rotation_head
        # self.active_head = self.rotation_head
        # self.num_classes = self.num_rotation_classes


def create_model(model_dir, model_choice, model_variant, num_classes=10, num_rotation_classes=4, classification_layers=512, rotation_layers=512):
    model_module_path = model_dir+"/"+model_choice+".py"
    model_module      = importlib.util.spec_from_file_location("",model_module_path).loader.load_module()
    model_function    = getattr(model_module, model_variant)
    model             = model_function(num_classes=num_classes, pretrained=False)
    # multi_task_model  = MultiTaskResNet(model, num_classes, num_rotation_classes, classification_layers=classification_layers, rotation_layers=rotation_layers)
    return model

def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model

def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def model_selection(model_selection_flag=0, model_dir="", model_choice="", model_variant="", saved_model_filepath="",num_classes=1, device="", mutli_selection_flag = True):
    if model_selection_flag == 0:
        # Create an untrained model.
        model = create_model(model_dir, model_choice, model_variant, num_classes)

    elif model_selection_flag == 1 and model_variant=="resnet26":
        # Load a pretrained model from Pytorch.
        # print("CUSTOM LOAD")
        model = create_model(model_dir, model_choice, model_variant, num_classes)
        state_dict = torch.hub.load_state_dict_from_url("https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth", progress=True)
        model.load_state_dict(state_dict)

    elif model_selection_flag == 1:
        # Load a pretrained model from Pytorch.
        model = torch.hub.load('pytorch/vision:v0.10.0', model_variant, pretrained=True)
        # Convert to a multi task model
        model  = MultiTaskResNet(model, num_classes)

    elif model_selection_flag == 2:
        # Load a local pretrained model.
        model = create_model(model_dir, model_choice, model_variant, num_classes)
        model = load_model(model=model, model_filepath=saved_model_filepath, device=device)
        if mutli_selection_flag:
            model  = MultiTaskResNet(model, num_classes)

    return model


def train_model(model, train_loader, test_loader, device, learning_rate=1e-2, num_epochs=200 ):

    classification_criterion = nn.CrossEntropyLoss()
    rotation_criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # print(model)
    for epoch in range(num_epochs):

        # Training
        model.train()

        total_classification_loss = 0.0
        total_rotation_loss = 0.0

        # running_loss = 0
        # running_corrects = 0

        for inputs, labels, rotation_labels in train_loader:
            # print("Model training..")
            inputs = inputs.to(device)
            labels = labels.to(device)
            rotation_labels = rotation_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # classification_output, rotation_output = model(inputs)
            classification_output,rotation_output = model(inputs)
            # print(outputs)
            # input("I have gone through outputs now")
            # classification_output = outputs 
            # rotation_output = outputs

            # Compute classification loss and rotation classification loss
            classification_loss = classification_criterion(classification_output, labels)
      

            # print(torch.argmax(inputs[:, 0, :, :], dim=1)) STOPEED AT "Modify dataloader to provide rotations and rotation labels. Current error being faced is because of an issue with rotation labels"
            rotation_loss = rotation_criterion(rotation_output, rotation_labels)
        

            # Combine the losses with weights (you can adjust these weights as needed)
            total_loss = classification_loss + 0.5 * rotation_loss

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()

            # forward + backward + optimize
            # outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            # statistics
            # running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)
            
            total_classification_loss += classification_loss.item()
            total_rotation_loss += rotation_loss.item()
     
            # for i in range(len(inputs)):

            #     image = inputs[i].permute(1, 2, 0)  # Rearrange channels for plotting (C, H, W) to (H, W, C)
            #     label = labels[i].item()
            #     rotation_label = rotation_labels[i].item()

            #     image = image.cpu()
            #     # Plot and save the image
            #     plt.figure(figsize=(3, 3))
            #     plt.imshow(image)
            #     plt.title(f"Class: {label}, Rotation: {rotation_label * 90}°")
            #     plt.axis('off')
            #     plt.savefig(f"image_{label}_{rotation_label}.png")  # Save the image with a unique filename

        # print(f'Epoch [{epoch + 1}/{num_epochs}] Classification Loss: {total_classification_loss:.4f}, Rotation Loss: {total_rotation_loss:.4f}')
        # train_loss = running_loss / len(train_loader.dataset)
        # train_accuracy = running_corrects / len(train_loader.dataset)

        # # Evaluation
        # model.eval()
        eval_loss, eval_accuracy, eval_rot_accuracy = top1Accuracy_rotation(model=model, test_loader=test_loader, device=device, criterion=None)

        print("Epoch: {:02d} Eval Acc Clas: {:.3f} Eval Acc Rot: {:.3f}".format(epoch, eval_accuracy, eval_rot_accuracy))

    return model