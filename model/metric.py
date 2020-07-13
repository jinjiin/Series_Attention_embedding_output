import torch


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def smape(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true) / (torch.abs(y_pred) + torch.abs(y_true))) * 2
    # return torch.mean(torch.abs(y_pred - y_true) / (y_pred + y_true)) * 2
