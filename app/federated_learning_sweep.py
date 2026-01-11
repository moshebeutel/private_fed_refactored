import argparse
import json
import logging
from functools import partial
from pathlib import Path
import wandb
import private_federated
import private_federated.common
from app.utils import create_run_name
from private_federated.common import builder
from private_federated.common import utils
from private_federated.data.dataset_factory import DatasetFactory
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
        args.dataset_name = config.dataset_name
        args.saved_models_path = Path.home() / 'saved_models' / config.model_name / config.dataset_name

        args.num_clients_agg = config.num_clients_agg
        args.num_clients_private = config.num_private_clients
        args.num_clients_public = config.num_clients_public
        DatasetFactory.CLASSES_PER_USER = args.num_clients_public
        args.classes_per_user = config.classes_per_user
        args.noise_multiplier = config.noise_multiplier
        args.clip = config.clip
        args.use_gp = config.use_gp
        args.embed_grads = config.embed_grads
        args.client_learning_rate = config.client_learning_rate
        args.server_learning_rate = config.server_learning_rate
        args.clients_internal_epochs = config.clients_internal_epochs
        if args.embed_grads:
            args.embedding_num_bases = config.gep_num_bases
            args.grads_history_size = config.grads_history_size

        create_run_name(args)
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

    sweep_config = {'name': args.sweep_name,
                    'method': 'grid',
                    'metric': {'name': 'best_epoch_validation_acc', 'goal': 'maximize'},
                    'parameters': parameters_dict}

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    wandb.agent(sweep_id, partial(sweep_train, sweep_id=sweep_id, args=args))


if __name__ == '__main__':
    run_sweep()
