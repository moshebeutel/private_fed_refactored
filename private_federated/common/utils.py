import logging
from pathlib import Path

import torch

from private_federated.differential_privacy.gep.gep_server import GepServer
from private_federated.federated_learning.clients_factory import ClientFactory
from private_federated.common.config import Config, to_dict
from private_federated.data.dataset_factory import DatasetFactory
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.federated_learning.client import Client
from private_federated.federated_learning.server import Server
from private_federated.models.model_factory import ModelFactory


def populate_args(args):

    assert args.model_name in ModelFactory.MODEL_HUB.keys(), (f'Expected one of {ModelFactory.MODEL_HUB.keys()}.'
                                                              f' Got {args.model_name}')
    Config.MODEL_NAME = args.model_name

    DatasetFactory.DATASETS_DIR = args.data_path
    DataLoadersGenerator.CLASSES_PER_USER = args.classes_per_user
    DataLoadersGenerator.BATCH_SIZE = args.batch_size

    Client.INTERNAL_EPOCHS = args.clients_internal_epochs
    Client.OPTIMIZER_PARAMS['lr'] = args.client_learning_rate

    # ClientFactory.NUM_ALL_USERS = args.num_clients_total
    ClientFactory.NUM_CLIENTS_PRIVATE = args.num_clients_private
    ClientFactory.NUM_CLIENTS_PUBLIC = args.num_clients_public


    Server.NUM_ROUNDS = args.num_rounds
    Server.NUM_CLIENT_AGG = args.num_clients_agg
    Server.SAMPLE_CLIENTS_WITH_REPLACEMENT = args.sample_with_replacement
    Server.LEARNING_RATE = args.server_learning_rate
    Server.WEIGHT_DECAY = args.weight_decay
    Server.MOMENTUM = args.momentum

    GepServer.NUM_BASIS_ELEMENTS = args.embedding_num_bases
    GepServer.GRADIENTS_HISTORY_SIZE = args.grads_history_size

    Config.USE_GP = args.use_gp
    Config.EMBED_GRADS = args.embed_grads
    Config.CLIP_VALUE = args.clip
    Config.NOISE_MULTIPLIER = args.noise_multiplier
    Config.DEVICE = torch.device(
            "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
        )

    logging.info({**to_dict(ModelFactory),
                  **to_dict(DataLoadersGenerator),
                  **to_dict(Client),
                  **to_dict(ClientFactory),
                  **to_dict(Server),
                  **to_dict(GepServer),
                  **to_dict(Config)})


def split_to_floats(inp: str) -> list[float]:
    lstrings = inp.split(sep=',')
    return [float(entry) for entry in lstrings]


def get_command_line_arguments(parser):
    """
    Parse command-line arguments.

    Parameters:
    parser (argparse.ArgumentParser): Argument parser object

    Returns:
    argparse.Namespace: Parsed arguments
    """
    # Data
    parser.add_argument("--data-path", type=str, default=f"{str(Path.home())}/datasets/",
                        help="dir path for datafolder")
    parser.add_argument("--dataset-name", type=str, default=f"CIFAR10",
                        choices=[DatasetFactory.DATASETS_HUB.keys()],
                        help="Name of dataset (CIFAR10, CIFAR100 ...)")

    # Federated Learning
    parser.add_argument("--num-clients-total", type=int, default=ClientFactory.NUM_ALL_USERS,
                        help="Number of clients in federation")
    parser.add_argument("--num-clients-agg", type=int, default=Server.NUM_CLIENT_AGG,
                        help="Number of clients sampled each round")
    parser.add_argument("--clients-internal-epochs", type=int, default=Client.INTERNAL_EPOCHS,
                        help="Number of epochs each sampled client"
                             " preform internally before"
                             " returning grads")

    parser.add_argument("--classes_per_user", type=int, default=DataLoadersGenerator.CLASSES_PER_USER,
                        help="Number of data classes each user knows")
    parser.add_argument("--sample-with-replacement", type=bool, default=Server.SAMPLE_CLIENTS_WITH_REPLACEMENT,
                        help="Sampling with or without replacement")

    # Train Hyperparameter
    parser.add_argument("--num-rounds", type=int, default=Server.NUM_ROUNDS,
                        help="Number of federated_learning training rounds in federation")
    parser.add_argument("--batch-size", type=int, default=DataLoadersGenerator.BATCH_SIZE,
                        help="Number of images in train batch")
    parser.add_argument("--server-learning-rate", type=float, default=Server.LEARNING_RATE,
                        help="Gradients update factor each round")
    parser.add_argument("--client-learning-rate", type=float, default=Client.OPTIMIZER_PARAMS['lr'],
                        help="Gradients update factor each internal round")
    parser.add_argument("--weight-decay", type=float, default="1e-3", help="Optimizer weight decay parameters")
    parser.add_argument("--momentum", type=float, default="0.9", help="Optimizer momentum parameter")

    parser.add_argument("--model-name", type=str, choices=ModelFactory.get_model_hub_names(),
                        default=Config.MODEL_NAME,
                        help='network model name (resnet20 ...)')

    parser.add_argument("--use-cuda", type=bool, default=True,
                        help='Use GPU. Use cpu if not')

    # GP
    parser.add_argument("--use-gp", type=bool, default=Config.USE_GP,
                        help='Use Gaussian Process model on client side.')


    # GEP
    parser.add_argument("--embed-grads", action='store_true', help='Use GEP')
    parser.add_argument("--num-clients-public", type=int,
                        default=ClientFactory.NUM_CLIENTS_PUBLIC,
                        help="Number of public clients")
    parser.add_argument("--num-clients-private", type=int,
                        default=ClientFactory.NUM_CLIENTS_PRIVATE,
                        help="Number of private clients")
    parser.add_argument("--embedding-num-bases", type=int, default=GepServer.NUM_BASIS_ELEMENTS,
                        help="Number of embedding subspace basis elements")
    parser.add_argument("--grads-history-size", type=int, default=GepServer.GRADIENTS_HISTORY_SIZE,
                        help="Gradients history buffer size")

    # DP
    parser.add_argument("--clip", type=float, default=Config.CLIP_VALUE,
                        help='Gradients clip value. If inf - no clip')

    parser.add_argument("--noise-multiplier", type=float, default=Config.NOISE_MULTIPLIER,
                        help='ratio  (DP noise ratio) / sensitivity.'
                             ' 0.0 for non-private mechanisms.')

    parser.add_argument("--saved-models-path", type=str, default='./saved_models',
                        help='Train model in a federated_learning manner before fine tuning')

    args = parser.parse_args()
    return args
