import argparse
import private_federated.common.utils as common_utils
from app import federated_learning, federated_learning_sweep

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Private Federated Learning Sweep")
    args = common_utils.get_command_line_arguments(parser)
    # federated_learning_sweep.run_sweep(args)
    federated_learning.main(args)

