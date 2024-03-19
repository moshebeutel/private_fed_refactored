import logging

import torch.nn
from torch import nn

from private_federated.common.config import Config
from private_federated.models.resnet_cifar import resnet20, resnet32, resnet14, resnet8, resnet44


class ModelFactory:
    MODEL_HUB = {'resnet8': resnet8,
                 'resnet14': resnet14,
                 'resnet20': resnet20,
                 'resnet32': resnet32,
                 'resnet44': resnet44}

    @staticmethod
    def get_model_hub_names():
        return ModelFactory.MODEL_HUB.keys()

    @staticmethod
    def init_model_weights(model: nn.Module) -> nn.Module:
        for m in model.modules():
            if (isinstance(m, nn.Conv3d) or
                    isinstance(m, nn.Conv2d) or
                    isinstance(m, nn.Conv1d) or
                    isinstance(m, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        return model

    def __init__(self, model_name: str):
        assert model_name in self.MODEL_HUB.keys(), 'Unknown model name: {}'.format(model_name)
        self.model_fn = ModelFactory.MODEL_HUB[model_name]

    def get_model(self) -> nn.Module:
        model = self.model_fn()
        model = ModelFactory.init_model_weights(model)
        device = Config.DEVICE
        model.to(device)
        logging.debug(f'Created model: {self.model_fn} in device: {device}')
        return model
