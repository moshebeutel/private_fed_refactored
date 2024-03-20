import torch.nn
from private_federated.common.config import Config
from private_federated.models.model_factory import ModelFactory


@torch.no_grad()
def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    cloned_model = ModelFactory(Config.MODEL_NAME).get_model()
    cloned_model.load_state_dict(model.state_dict())
    return cloned_model


@torch.no_grad()
def merge_model(model1: torch.nn.Module,
                model2: torch.nn.Module,
                weight1: float = 0.5,
                weight2: float = 0.5) -> torch.nn.Module:
    cloned_model = ModelFactory(Config.MODEL_NAME).get_model()
    dict1 = model1.state_dict()
    dict2 = model2.state_dict()
    target_dict = cloned_model.state_dict()
    for key in dict1.keys():
        assert key in dict2.keys()
        assert dict1[key].shape == dict2[key].shape
        target_dict[key] = torch.clone(dict1[key] * weight1) + torch.clone(dict2[key] * weight2)
    cloned_model.load_state_dict(target_dict)
    return cloned_model


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
