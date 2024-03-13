import argparse
import logging
from functools import partial
import wandb
import private_federated
import private_federated.common
from private_federated.common import builder
from private_federated.common import utils
from private_federated.train.utils import set_seed


def single_train(args):
    private_federated.common.utils.populate_args(args)
    federated_learning_server = builder.build_all(args)
    federated_learning_server.federated_learn()


def sweep_train(sweep_id, args, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        logging.info(config)
        set_seed(config.seed)

        args.model_name = config.model_name
        args.num_clients_agg = config.num_clients_agg
        args.num_clients_private = config.num_private_clients
        args.classes_per_user = config.classes_per_user
        args.noise_multiplier = config.noise_multiplier
        args.clip = config.clip
        args.embed_grads = config.embed_grads
        args.num_clients_public = config.num_clients_public
        args.client_learning_rate = config.client_learning_rate
        args.server_learning_rate = config.server_learning_rate
        args.clients_internal_epochs = config.clients_internal_epochs

        run_name = (f'Model Name: {args.model_name},'
                    f'Num Clients Agg: {args.num_clients_agg}'
                    f'Noise Mult. {args.noise_multiplier},'
                    f'Clip Value {args.clip},'
                    f'Internal Epochs {args.clients_internal_epochs},'
                    f'Server Learning Rate {args.server_learning_rate},'
                    f'Client Learning Rate {args.client_learning_rate}')
        if args.embed_grads:
            args.embedding_num_bases = config.gep_num_bases

            run_name += (f','
                         f'Num Basis Elements {args.embedding_num_bases},'
                         f'Num Public Clients {args.num_clients_public}')

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
    parameters_dict = {
        'noise_multiplier': {
            'values': [25.0, 0.0]
            # 'values': [12.79182, 4.72193, 2.01643, 0.0]
        },
        'embed_grads': {
            'values': [False, True]
        },
        'num_clients_agg': {
            'values': [20]
        },
        'num_clients_public': {
            'values': [100]
        },
        'gep_num_bases': {
            'values': [80]
        },
        'clip': {
            'values': [1.0, 0.001]
        },
        'seed': {
            'values': [50]
        },
        'num_private_clients': {
            'values': [700]
        },
        'model_name': {
            'values': ['resnet20']
        },
        'classes_per_user': {
            'values': [2]
        },
        'clients_internal_epochs': {
            'values': [1, 5]
        },
        'client_learning_rate': {
            'values': [0.001, 1.0]
        },
        'server_learning_rate': {
            'values': [0.001, 1.0]
        }

    }

    parameters_dict.update({
        # 'sample_with_replacement': {
        #     'values': [0, 1]
        # },
        # 'use_gp': {
        #     'values': [0]
        # },
    })
    sweep_config['parameters'] = parameters_dict
    metric = {
        'name': 'best_epoch_validation_acc',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, partial(sweep_train, sweep_id=sweep_id, args=args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Private Federated Learning Sweep")
    command_line_args = private_federated.common.utils.get_command_line_arguments(parser)
    run_sweep(command_line_args)
