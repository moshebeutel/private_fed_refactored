import torch.nn
from private_federated.models.resnet_cifar import resnet8, resnet20


@torch.no_grad()
def clone_model(model: torch.nn.Module, include_grads: bool = False) -> torch.nn.Module:
    cloned_model = resnet20()
    for source_param, target_param in zip(model.parameters(), cloned_model.parameters()):
        target_param = torch.zeros_like(source_param, device=source_param.device)
        target_param.data = source_param.data
        if include_grads:
            assert source_param.grad is not None, "source_param has no gradient"
            assert target_param.grad is not None, "target_param has no gradient"
            target_param.grad = torch.zeros_like(source_param.grad, device=source_param.grad.device)
            target_param.grad.data = source_param.grad.data
    return cloned_model.to(next(model.parameters()).device)


@torch.no_grad()
def merge_model(model1: torch.nn.Module,
                model2: torch.nn.Module,
                include_grads: bool = False,
                weight1: float = 0.5,
                weight2: float = 0.5) -> torch.nn.Module:
    cloned_model = resnet20()
    for source_param1, source_param2, target_param in zip(model1.parameters(), model2.parameters(),
                                                          cloned_model.parameters()):
        target_param = torch.zeros_like(source_param1, device=source_param1.device)
        target_param.data = source_param1.data * weight1 + source_param2 + weight2
        if include_grads:
            assert source_param1.grad is not None, "source_param has no gradient"
            assert source_param2.grad is not None, "source_param has no gradient"
            target_param.grad = torch.zeros_like(source_param1.grad, device=source_param1.grad.device)
            target_param.grad.data = source_param1.grad.data * weight1 + source_param2 * weight2
    return cloned_model.to(next(model1.parameters()).device)


@torch.no_grad()
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
