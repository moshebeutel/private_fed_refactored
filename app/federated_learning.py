import wandb

import private_federated.common
from common.config import Config
from private_federated.common import builder
from private_federated.common import utils


def main(args):
    private_federated.common.utils.populate_args(args)
    federated_learning_server = builder.build_all(args)
    if Config.LOG2WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name='federated simple')
    federated_learning_server.federated_learn()


