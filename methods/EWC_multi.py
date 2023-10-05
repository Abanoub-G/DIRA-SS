import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from metrics.accuracy.topAccuracy import top1Accuracy

def on_task_update(task_id, trainloader, model, optimizer, fisher_dict, optpar_dict, device):

    model.train()
    optimizer.zero_grad()
    
    # accumulating gradients
    for inputs, labels, _ in trainloader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs,_ = model(inputs)
    
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

    fisher_dict[task_id] = {}
    optpar_dict[task_id] = {}

    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
        
        optpar_dict[task_id][name] = param.data.clone()
        fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

    return fisher_dict, optpar_dict


def train_model_ewc(model, train_loader, test_loader, device, fisher_dict, optpar_dict, num_epochs=200, ewc_lambda = 1, learning_rate=1e-2, momentum=0.9, weight_decay=1e-5 ):

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    
    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels, _ in train_loader:
            # print("Model training..")
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # print("inputs type = ",inputs.dtype)
            # print("outputs type = ",outputs.dtype)
            # print("inputs shape = ",inputs.shape)
            # print("outputs shape = ",outputs.shape)

            # print("labels type = ",labels.dtype)
            # print("labels shape = ",labels.shape)
            # input("press etner 9")
            

            # print(inputs)
            # input("Press enter")
            # print(outputs)
            # input("press enter")
            loss = criterion(outputs, labels)
            
            # EWC -- magic here! :-)
            task_id = 1
            for task in range(task_id):
                # print("I am in EWC and I am working on taks nummber", task)
                # print("ewc_lambda[task] = ",ewc_lambda[task])
                for name, param in model.named_parameters():
                    fisher = fisher_dict[task][name]
                    optpar = optpar_dict[task_id-1][name]
                    loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
                    # loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda[task]
                    # loss += ((optpar - param).pow(2)).sum() * ewc_lambda   


            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = top1Accuracy(model=model, test_loader=test_loader, device=device, criterion=criterion)

        print("Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy))

    return model
