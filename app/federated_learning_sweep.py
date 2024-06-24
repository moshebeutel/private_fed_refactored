import argparse
import logging
import sys
from functools import partial
import wandb
import private_federated
import private_federated.common
from private_federated.common import builder
from private_federated.common import utils
from private_federated.train.utils import set_seed
from pathlib import Path
import json
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
        args.dataset_name = config.dataset_name
        args.num_clients_agg = config.num_clients_agg
        args.num_clients_private = config.num_private_clients
        args.num_clients_public = config.num_clients_public
        args.classes_per_user = config.classes_per_user
        args.noise_multiplier = config.noise_multiplier
        args.clip = config.clip
        args.use_gp = config.use_gp
        args.embed_grads = config.embed_grads
        args.client_learning_rate = config.client_learning_rate
        args.server_learning_rate = config.server_learning_rate
        args.clients_internal_epochs = config.clients_internal_epochs

        run_name = (f'Use GP: {args.use_gp},'
                    f'Embed Grads: {args.embed_grads},'
                    f'Num Clients Agg: {args.num_clients_agg},'
                    f'Noise Mult. {args.noise_multiplier},'
                    f'Clip Value {args.clip},'
                    f'Internal Epochs {args.clients_internal_epochs},'
                    f'Server Learning Rate {args.server_learning_rate},'
                    f'Client Learning Rate {args.client_learning_rate}')
        if args.embed_grads:
            args.embedding_num_bases = config.gep_num_bases
            args.grads_history_size = config.grads_history_size

            run_name += (f','
                         f'Num Basis Elements {args.embedding_num_bases},'
                         f'Grads History Size {args.grads_history_size},'
                         f'Num Public Clients {args.num_clients_public}')

        logging.info(run_name)
        print('\n'.join(run_name.split(',')))
        wandb.run.name = run_name
        single_train(args)


def run_sweep():
    parser = argparse.ArgumentParser(description="Private Federated Learning Sweep")
    args = private_federated.common.utils.get_command_line_arguments(parser)

    logging.basicConfig(level=logging.INFO)
    logging.info("run sweep")

    json_path = Path(args.json_path)
    assert json_path.exists(), f'{json_path} does not exist'
    assert json_path.suffix == '.json', f'{json_path} is not a json file'

    with open(json_path, 'r') as file:
        json_data = file.read()

    # Convert JSON data to a dictionary
    parameters_dict = json.loads(json_data)

    sweep_config = {'method': 'grid', 'parameters': parameters_dict}

    metric = {
        'name': 'best_epoch_validation_acc',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, partial(sweep_train, sweep_id=sweep_id, args=args))


if __name__ == '__main__':
    run_sweep()
