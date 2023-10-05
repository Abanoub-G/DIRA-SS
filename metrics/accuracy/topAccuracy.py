import torch

def top1Accuracy(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def top1Accuracy_rotation(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0
    running_corrects_rotation = 0

    for inputs, labels, rotation_labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        rotation_labels = rotation_labels.to(device)

        classification_outputs,rotation_outputs = model(inputs)

        _, classification_preds = torch.max(classification_outputs, 1)
        _, rotation_preds = torch.max(rotation_outputs, 1)

        if criterion is not None:
            loss = criterion(classification_outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(classification_preds == labels.data)
        running_corrects_rotation += torch.sum(rotation_preds == rotation_labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    eval_rot_accuracy = running_corrects_rotation / len(test_loader.dataset)

    return eval_loss, eval_accuracy, eval_rot_accuracy

