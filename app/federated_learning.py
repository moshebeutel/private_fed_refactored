import argparse
import logging
import wandb
import private_federated
from private_federated.common import builder, utils, config


def run_single(args):
    logging.basicConfig(level=logging.INFO)
    logging.info("run single")
    private_federated.common.utils.populate_args(args)
    logging.info(f'Embed grads {args.embed_grads} Noise mult. {args.noise_multiplier} clip {args.clip}')
    federated_learning_server = builder.build_all(args)
    if config.Config.LOG2WANDB:
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name='federated simple')
    federated_learning_server.federated_learn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Private Federated Learning Run")
    args = utils.get_command_line_arguments(parser)
    run_single(args)
