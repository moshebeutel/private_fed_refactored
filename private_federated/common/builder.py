import logging
from functools import partial
import torch
from torch.utils.data import Dataset
from private_federated.aggregation_strategies.average_clip_strategy import AverageClipStrategy
from private_federated.aggregation_strategies.average_strategy import AverageStrategy
from private_federated.common.config import Config
from private_federated.data.dataset_factory import DatasetFactory
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.differential_privacy.dp_sgd.dp_sgd_aggregation_starategy import DpSgdAggregationStrategy
from private_federated.differential_privacy.gep.gep_server import GepServer
from private_federated.federated_learning.clients_factory import ClientFactory
from private_federated.federated_learning.server import Server
from private_federated.models import model_factory


def get_aggregation_strategy(args):
    if args.clip < float('inf'):
        assert args.clip > 0.0, f'Expected positive clip value. Got {args.clip}'
        if args.noise_multiplier > 0.0:
            strategy = DpSgdAggregationStrategy(clip_value=args.clip, noise_multiplier=args.noise_multiplier)
        else:
            strategy = AverageClipStrategy(clip_value=args.clip)
    else:
        assert args.noise_multiplier == 0.0, (f'No clip given. '
                                              f'Expected non-private but got noise multiplier {args.noise_multiplier}')
        strategy = AverageStrategy()
    logging.info(f'Created aggregation strategy {strategy}')
    assert strategy is not None, 'Some unexpected logic path'
    return strategy


def get_server_params(server_type_name,
                      clients_factory: ClientFactory,
                      models_factory,
                      dataset_factory,
                      aggregation_strategy_factory):
    clients = clients_factory.private_train_clients
    net = models_factory()
    server_test_loader, server_val_loader = get_loaders(dataset_factory)
    strategy = aggregation_strategy_factory()
    logging.info(f'Aggregation strategy: {strategy.__class__.__name__}')
    server_params = {'clients': clients,
                     'net': net,
                     'val_loader': server_val_loader,
                     'test_loader': server_test_loader,
                     'aggregating_strategy': strategy}

    if server_type_name == GepServer.__name__:
        server_params['private_clients'] = server_params.pop('clients')
        server_params['public_clients'] = clients_factory.public_clients
    return server_params


def get_server_type() -> str:
    return (Server if not Config.EMBED_GRADS else GepServer).__name__


def build_all(args):
    dataset_factory = DatasetFactory(dataset_name=args.dataset_name)
    clients_factory = ClientFactory(dataset_factory)
    models_factory_fn = partial(model_factory.get_model, args)
    aggregation_strategy_factory_fn = partial(get_aggregation_strategy, args)
    server = get_server(aggregation_strategy_factory_fn, clients_factory, dataset_factory, models_factory_fn)
    return server


def get_server(aggregation_strategy_factory_fn, clients_factory, dataset_factory, models_factory_fn):
    server_type_name = get_server_type()
    logging.info(f'Server type: {server_type_name}')
    server_params = get_server_params(server_type_name,
                                      clients_factory,
                                      models_factory_fn,
                                      dataset_factory,
                                      aggregation_strategy_factory_fn)
    server = globals()[server_type_name](**server_params)
    return server


def get_loaders(dataset_factory):
    loader_params = {"batch_size": DataLoadersGenerator.BATCH_SIZE, "shuffle": False,
                     "pin_memory": True, "num_workers": 0}
    server_val_loader = create_loader_from_dataset(dataset_factory.val_set, **loader_params)
    server_test_loader = create_loader_from_dataset(dataset_factory.test_set, **loader_params)
    return server_test_loader, server_val_loader


def create_loader_from_dataset(dataset: Dataset, **params):
    return torch.utils.data.DataLoader(dataset=dataset, **params)
