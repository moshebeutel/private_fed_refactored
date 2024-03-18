import logging
from functools import partial
from typing import Callable
import torch
from torch.utils.data import Dataset, DataLoader

from private_federated.models.resnet_cifar import resnet8, resnet20
from private_federated.aggregation_strategies.average_strategy import AverageStrategy
from private_federated.common.config import Config
from private_federated.data.dataset_factory import DatasetFactory
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.differential_privacy.dp_sgd.dp_sgd_aggregation_starategy import DpSgdAggregationStrategy
from private_federated.differential_privacy.gep.gep_server import GepServer
from private_federated.federated_learning.clients_factory import ClientFactory
from private_federated.federated_learning.gp_client_factory import GPClientFactory
from private_federated.federated_learning.server import Server



def get_aggregation_strategy(args):
    """
    Aggregation strategy factory method
    :param args: Command line arguments
    :return: aggregation strategy instance
    """
    if args.clip < float('inf'):
        assert args.clip > 0.0, f'Expected positive clip value. Got {args.clip}'
        strategy = DpSgdAggregationStrategy(clip_value=args.clip, noise_multiplier=args.noise_multiplier)
    else:
        assert args.noise_multiplier == 0.0, (f'No clip given. '
                                              f'Expected non-private but got noise multiplier {args.noise_multiplier}')
        strategy = AverageStrategy()
    logging.info(f'Created aggregation strategy {strategy}')
    assert strategy is not None, 'Some unexpected logic path'
    return strategy


def get_server_params(server_type_name: str,
                      clients_factory: ClientFactory,
                      models_factory_method: Callable[[], torch.nn.Module],
                      dataset_factory: DatasetFactory,
                      aggregation_strategy_factory_method: Callable[[], AverageStrategy]) -> dict:
    """
    Server parameters factory method
    :param server_type_name: Server type name
    :param clients_factory:  Client factory
    :param models_factory_method: Model factory method
    :param dataset_factory: Dataset factory
    :param aggregation_strategy_factory_method: Aggregation strategy factory method
    """
    clients = clients_factory.private_train_clients
    val_clients = clients_factory.validation_clients
    test_clients = clients_factory.test_clients
    net = models_factory_method()
    server_test_loader, server_val_loader = get_loaders(dataset_factory)
    strategy = aggregation_strategy_factory_method()
    logging.info(f'Aggregation strategy: {strategy.__class__.__name__}')
    server_params = {'train_clients': clients,
                     'val_clients': val_clients,
                     'test_clients': test_clients,
                     'net': net,
                     'val_loader': server_val_loader,
                     'test_loader': server_test_loader,
                     'aggregating_strategy': strategy}

    if server_type_name == GepServer.__name__:
        server_params['private_clients'] = server_params.pop('train_clients')
        server_params['public_clients'] = clients_factory.public_clients
    return server_params


def get_server_type() -> str:
    return (Server if not Config.EMBED_GRADS else GepServer).__name__


def get_clients_factory_type(args):
    return GPClientFactory if args.use_gp else ClientFactory


def build_all(args) -> Server:
    dataset_factory = DatasetFactory(dataset_name=args.dataset_name)
    clients_factory = get_clients_factory_type(args)(dataset_factory)
    models_factory_fn = resnet20
    aggregation_strategy_factory_fn = partial(get_aggregation_strategy, args)
    server: Server = get_server(aggregation_strategy_factory_fn, clients_factory, dataset_factory, models_factory_fn)
    return server


def get_server(aggregation_strategy_factory_fn, clients_factory: ClientFactory, dataset_factory: DatasetFactory,
               models_factory_fn: Callable[[], torch.nn.Module]) -> Server:
    server_type_name: str = get_server_type()
    logging.info(f'Server type: {server_type_name}')
    server_params = get_server_params(server_type_name,
                                      clients_factory,
                                      models_factory_fn,
                                      dataset_factory,
                                      aggregation_strategy_factory_fn)
    server = globals()[server_type_name](**server_params)
    return server


def get_loaders(dataset_factory: DatasetFactory) -> tuple[DataLoader, DataLoader]:
    loader_params: dict = {"batch_size": DataLoadersGenerator.BATCH_SIZE,
                           "shuffle": False,
                           "pin_memory": True,
                           "num_workers": 0}
    server_val_loader: DataLoader = create_loader_from_dataset(dataset_factory.val_set, **loader_params)
    server_test_loader: DataLoader = create_loader_from_dataset(dataset_factory.test_set, **loader_params)
    return server_test_loader, server_val_loader


def create_loader_from_dataset(dataset: Dataset, **params) -> DataLoader:
    return torch.utils.data.DataLoader(dataset=dataset, **params)
