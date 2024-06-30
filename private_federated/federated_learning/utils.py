import logging

from private_federated.federated_learning.client import Client


def evaluate_clients(clients):
    total_accuracy, total_loss = 0.0, 0.0
    for c in clients:
        acc, loss = c.evaluate()
        total_accuracy += acc
        total_loss += loss
    logging.debug(f'evaluate_clients acc {total_accuracy / float(len(clients))}')
    return total_accuracy / float(len(clients)), total_loss / float(len(clients))
