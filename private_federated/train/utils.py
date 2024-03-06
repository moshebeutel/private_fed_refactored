import torch.nn


def evaluate(net, loader, criterion) -> tuple[float, float]:
    eval_accuracy, total, eval_loss = 0.0, 0.0, 0.0
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        # for data in tqdm(test_loader):
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels).item()
            eval_loss += float(loss)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            eval_accuracy += (predicted == labels).sum().item()
            del predicted, loss, images, labels
    eval_accuracy /= total
    eval_loss /= total
    return eval_accuracy, eval_loss


def set_seed(seed, cudnn_enabled=True):
    import numpy as np
    import random
    import torch

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
