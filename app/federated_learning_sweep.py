import argparse
import logging
from functools import partial
import wandb
import private_federated
import private_federated.common
from private_federated.common import builder
from private_federated.common import utils
from private_federated.common.config import Config
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.differential_privacy.gep.gep_server import GepServer
from private_federated.federated_learning.clients_factory import ClientFactory


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


def single_train(args):
    private_federated.common.utils.populate_args(args)
    federated_learning_server = builder.build_all(args)
    federated_learning_server.federated_learn()


def sweep_train(sweep_id, args, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        logging.info(config)
        set_seed(30)
        args.classes_per_user = config.classes_per_user
        DataLoadersGenerator.CLASSES_PER_USER = config.classes_per_user
        args.noise_multiplier = config.noise_multiplier
        Config.NOISE_MULTIPLIER = config.noise_multiplier
        args.clip = config.clip
        Config.CLIP_VALUE = config.clip
        args.embed_grads = config.embed_grads
        Config.EMBED_GRADS = config.embed_grads
        GepServer.NUM_BASIS_ELEMENTS = config.gep_num_bases
        args.embedding_num_bases = config.gep_num_bases
        args.num_clients_public = config.num_clients_public
        ClientFactory.NUM_CLIENTS_PUBLIC = config.num_clients_public
        run_name = f'Embed grads {args.embed_grads},Noise mult. {args.noise_multiplier},Clip {args.clip},Num Basis {args.embedding_num_bases}, Num Public {args.num_clients_public}'
        # run_name = f'Embed grads {args.embed_grads},Noise mult. {args.noise_multiplier},Clip {args.clip}'
        logging.info(run_name)
        print('\n'.join(run_name.split(',')))
        wandb.run.name = run_name
        single_train(args)


def run_sweep(args):
    logging.basicConfig(level=logging.INFO)
    logging.info("run sweep")
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {}

    sweep_config['parameters'] = parameters_dict
    metric = {
        'name': 'best_epoch_validation_acc',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    parameters_dict.update({
        'noise_multiplier': {
            'values': [12.79182, 4.72193, 2.01643, 0.0]
        },
        'embed_grads': {
            'values': [True]
        },
        'clip': {
            'values': [0.001, 0.1]
        },
        # 'seed': {
        #     'values': [20]
        #     # 'values': [20, 40, 60]
        # },
        # 'sample_with_replacement': {
        #     'values': [0, 1]
        # },
        # 'num_clients_agg': {
        #     'values': [50]
        #     # 'values': [10, 50]
        # },
        'num_clients_public': {
            'values': [10, 100]
        },
        'classes_per_user': {
            'values': [2, 10]
        },
        # 'clients_internal_epochs': {
        #     'values': [1, 5]
        # },
        # 'use_gp': {
        #     'values': [0]
        # },
        'gep_num_bases': {
            'values': [10, 100]
        }
    })

    # parameters_dict.update({'epsilon': {'values': split_to_floats(args.epsilon_values)}})

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, partial(sweep_train, sweep_id=sweep_id, args=args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Private Federated Learning Sweep")
    args = private_federated.common.utils.get_command_line_arguments(parser)
    run_sweep(args)
