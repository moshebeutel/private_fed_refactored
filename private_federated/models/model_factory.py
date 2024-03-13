import logging
import torch.nn
from private_federated.models.resnet_cifar import resnet20, resnet32, resnet14, resnet8, resnet44
from torch import nn


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

    @staticmethod
    def get_model(args) -> nn.Module:
        assert args.model_name in ModelFactory.MODEL_HUB.keys(), (f'Expected one of {ModelFactory.MODEL_HUB.keys()}.'
                                                                  f' Got {args.model_name}')
        model = ModelFactory.MODEL_HUB[args.model_name]()
        model = ModelFactory.init_model_weights(model)
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
        )
        model.to(device)
        logging.info(f'Created model: {args.model_name} in device: {device}')
        return model
