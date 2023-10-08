from pathlib import Path
from private_federated.data.dataset_factory import DatasetFactory
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.federated_learning.client import Client
from private_federated.federated_learning.server import Server
from private_federated.models.model_factory import get_model_hub_names


def populate_args(args):

    DatasetFactory.DATASETS_DIR = args.data_path
    DataLoadersGenerator.CLASSES_PER_USER = args.classes_per_user
    DataLoadersGenerator.BATCH_SIZE = args.batch_size

    Client.INTERNAL_EPOCHS = args.clients_internal_epochs

    Server.NUM_ROUNDS = args.num_rounds
    Server.NUM_CLIENT_AGG = args.num_clients_agg
    Server.SAMPLE_CLIENTS_WITH_REPLACEMENT = args.sample_with_replacement
    Server.LEARNING_RATE = args.learning_rate
    Server.WEIGHT_DECAY = args.weight_decay
    Server.MOMENTUM = args.momentum


def get_command_line_arguments(parser):
    """
    Parse command-line arguments.

    Parameters:
    parser (argparse.ArgumentParser): Argument parser object

    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser.add_argument("--data-path", type=str, default=f"{str(Path.home())}/datasets/",
                        help="dir path for datafolder")
    parser.add_argument("--dataset-name", type=str, default=f"cifar10",
                        choices=[DatasetFactory.DATASETS_HUB.keys()],
                        help="Name of dataset (cifar10, cifar100 ...)")
    parser.add_argument("--num-clients", type=int, default="500", help="Number of clients in federation")
    parser.add_argument("--num-clients-agg", type=int, default=Server.NUM_CLIENT_AGG,
                        help="Number of clients sampled each round")
    parser.add_argument("--clients-internal-epochs", type=int, default=Client.INTERNAL_EPOCHS,
                        help="Number of epochs each sampled client"
                             " preform internally before"
                             " returning grads")
    parser.add_argument("--num-rounds", type=int, default=Server.NUM_ROUNDS,
                        help="Number of federated_learning training rounds in federation")
    parser.add_argument("--batch-size", type=int, default=DataLoadersGenerator.BATCH_SIZE,
                        help="Number of images in train batch")
    parser.add_argument("--classes_per_user", type=int, default=DataLoadersGenerator.CLASSES_PER_USER,
                        help="Number of data classes each user knows")
    parser.add_argument("--sample-with-replacement", type=bool, default=Server.SAMPLE_CLIENTS_WITH_REPLACEMENT,
                        help="Sampling with or without replacement")

    parser.add_argument("--learning-rate", type=float, default=Server.LEARNING_RATE,
                        help="Gradients update factor each round")
    parser.add_argument("--weight-decay", type=float, default="1e-3", help="Optimizer weight decay parameters")
    parser.add_argument("--momentum", type=float, default="0.9", help="Optimizer momentum parameter")

    parser.add_argument("--model-name", type=str, choices=get_model_hub_names(), default='resnet20',
                        help='network model name (resnet20 ...)')

    parser.add_argument("--load-from", type=str,
                        default='',
                        help='Load a pretrained model from given path. Train from scratch if string empty')

    parser.add_argument("--preform-pretrain",
                        type=bool,
                        default=True,
                        help='Train model in a federated_learning manner before fine tuning')

    parser.add_argument("--use-cuda", type=bool, default=True,
                        help='Use GPU. Use cpu if not')

    parser.add_argument("--saved-models-path", type=str, default='./saved_models',
                        help='Train model in a federated_learning manner before fine tuning')
    args = parser.parse_args()
    return args
