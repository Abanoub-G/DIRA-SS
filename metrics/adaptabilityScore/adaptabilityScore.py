import torch

def adaptabilityScore(N_d, N_T, A_T)#(model, test_loader, device, criterion=None):

    return A_T * N_d / N_T
    # model.eval()
    # model.to(device)

    # running_loss = 0
    # running_corrects = 0

    # for inputs, labels in test_loader:

    #     inputs = inputs.to(device)
    #     labels = labels.to(device)

    #     outputs = model(inputs)
    #     _, preds = torch.max(outputs, 1)

    #     if criterion is not None:
    #         loss = criterion(outputs, labels).item()
    #     else:
    #         loss = 0

    #     # statistics
    #     running_loss += loss * inputs.size(0)
    #     running_corrects += torch.sum(preds == labels.data)

    # eval_loss = running_loss / len(test_loader.dataset)
    # eval_accuracy = running_corrects / len(test_loader.dataset)

    # return eval_loss, eval_accuracy

