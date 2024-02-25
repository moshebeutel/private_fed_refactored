import logging
import torch.nn
from private_federated.models.resnet_cifar import resnet20
from torch import nn

model_hub = {'resnet20': resnet20}


def get_model_hub_names():
    return model_hub.keys()


def init_model(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def get_model(args):
    assert args.model_name in model_hub.keys(), f'Expected one of {model_hub.keys()}. Got {args.model_name}'
    model = model_hub[args.model_name]()
    model = init_model(model)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )
    model.to(device)
    logging.info(f'Created model: {args.model_name} in device: {device}')
    return model
