import wandb
from private_federated.common import builder
import private_federated.common
from private_federated.common import utils


def main(args):
    private_federated.common.utils.populate_args(args)
    wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name='federated simple')
    federated_learning_server = builder.build_all(args)
    federated_learning_server.federated_learn()
