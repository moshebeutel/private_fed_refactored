import argparse
from functools import partial
import wandb
import private_federated.common
from private_federated.common.config import Config
from private_federated.differential_privacy.dp_sgd.dp_sgd_aggregation_starategy import DpSgdAggregationStrategy
from private_federated import common
from private_federated.aggregation_strategies.average_clip_strategy import AverageClipStrategy
from private_federated.common import builder
from private_federated.common import utils


def single_train(args):
    private_federated.common.utils.populate_args(args)
    federated_learning_server = builder.build_all(args)
    federated_learning_server.federated_learn()


def sweep_train(sweep_id, args, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        print(config)

        # args.classes_per_user = config.classes_per_user
        # args.batch_size = config.batch_size
        # args.clients_internal_epochs = config.clients_internal_epochs
        # args.num_rounds = config.num_rounds
        # args.num_clients_agg = config.num_clients_agg
        # args.learning_rate = config.learning_rate
        args.noise_multiplier = config.noise_multiplier
        DpSgdAggregationStrategy.NOISE_MULTIPLIER = config.noise_multiplier
        args.clip = config.clip
        AverageClipStrategy.CLIP_VALUE = config.clip
        args.embed_grads = config.embed_grads
        Config.EMBED_GRADS = config.embed_grads
        wandb.run.name = f'noise multiplier {args.noise_multiplier} clip {args.clip}'
        single_train(args)


def run_sweep(args):
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
        # 'learning_rate': {
        #     'values': [0.00001, 0.0001, 0.001, 0.01, 0.1]
        # },
        # 'batch_size': {
        #     'values': [128, 256, 512]
        # },
        # 'num_rounds': {
        #     'values': [100, 500, 1000]
        # },
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
    args = common.utils.get_command_line_arguments(parser)
    run_sweep(args)

