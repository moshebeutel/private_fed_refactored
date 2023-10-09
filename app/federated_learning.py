import wandb
from private_federated.common import builder


def main(args):
    wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name='federated simple')
    federated_learning_server = builder.build_all(args)
    federated_learning_server.federated_learn()
