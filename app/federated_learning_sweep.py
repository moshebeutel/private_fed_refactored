import argparse
import logging
from functools import partial
import wandb
import private_federated
import private_federated.common
from private_federated.common import builder
from private_federated.common import utils
from private_federated.common.config import Config


def single_train(args):
    private_federated.common.utils.populate_args(args)
    federated_learning_server = builder.build_all(args)
    federated_learning_server.federated_learn()


def sweep_train(sweep_id, args, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        logging.info(config)
        args.noise_multiplier = config.noise_multiplier
        Config.NOISE_MULTIPLIER = config.noise_multiplier
        args.clip = config.clip
        Config.CLIP_VALUE = config.clip
        args.embed_grads = config.embed_grads
        Config.EMBED_GRADS = config.embed_grads
        run_name = f'Embed grads {args.embed_grads} Noise mult. {args.noise_multiplier} clip {args.clip}'
        logging.info(run_name)
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
            'values': [12.79182, 4.72193, 2.01643]
        },
        'embed_grads': {
            'values': [True, False]
        },
        'clip': {
            'values': [0.001, 0.01, 0.1, 1.0]
        },
        # 'sigma': {
        #     'values': [1.2, 3.2, 9.6, 0.6, 1.6, 4.8]
        # },
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
        # 'dp': {
        #     # 'values': ['GEP_NO_RESIDUALS', 'GEP_RESIDUALS', 'SGD_DP', 'NO_DP']
        # },
        # 'num_clients_public': {
        #     'values': [150]
        #     # 'values': [25, 50, 70, 100]
        # },
        # 'classes_per_user': {
        #     'values': [2, 6, 10]
        # },
        # 'clients_internal_epochs': {
        #     'values': [1, 5]
        # },
        # 'use_gp': {
        #     'values': [0]
        # },
        # 'gep_num_bases': {
        #     'values': [150]
        # }
    })

    # parameters_dict.update({'epsilon': {'values': split_to_floats(args.epsilon_values)}})

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, partial(sweep_train, sweep_id=sweep_id, args=args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Private Federated Learning Sweep")
    args = private_federated.common.utils.get_command_line_arguments(parser)
    run_sweep(args)

