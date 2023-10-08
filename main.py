import argparse
import private_federated.common.utils as common_utils
from app import federated_learning

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Private Federated Learning Flower")
    args = common_utils.get_command_line_arguments(parser)
    common_utils.populate_args(args)
    federated_learning.main(args)

