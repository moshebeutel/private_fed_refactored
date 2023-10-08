from private_federated.common import builder


def main(args):
    federated_learning_server = builder.build_all(args)
    federated_learning_server.federated_learn()
