import torch.nn
from torch.nn import CrossEntropyLoss


def evaluate(net, loader, criterion):
    eval_accuracy, total, eval_loss = 0, 0, 0.0
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


def get_net_parameters(net: torch.nn.Module):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def init_net_grads(net: torch.nn.Module, value: float = 0.0):
    for p in net.parameters():
        p.grad = torch.ones_like(p) * value


def get_net_grads(net: torch.nn.Module):
    assert next(net.parameters()).grad is not None, f'Expected grads initiated'
    return {name: param.grad.data for (name, param) in net.named_parameters()}
