import logging
import random
import copy
from typing import Callable
import torch.nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from private_federated.common.config import Config
from private_federated.differential_privacy.gep.utils import flatten_tensor
from private_federated.federated_learning.client import Client
from private_federated.train.utils import evaluate
from private_federated.models.utils import get_net_grads, zero_net_grads


class Server:
    NUM_ROUNDS = 30
    NUM_CLIENT_AGG: int = 20
    SAMPLE_CLIENTS_WITH_REPLACEMENT: bool = True
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-3
    MOMENTUM: float = 0.9

    def __init__(self,
                 train_clients: list[Client],
                 val_clients: list[Client],
                 test_clients: list[Client],
                 net: torch.nn.Module,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 aggregating_strategy: Callable[[torch.tensor], torch.tensor]):
        self._train_clients: list[Client] = train_clients
        self._val_clients: list[Client] = val_clients
        self._test_clients: list[Client] = test_clients
        zero_net_grads(net)
        self._grads: {str: torch.tensor} = get_net_grads(net)
        self._shapes = [g.shape for g in self._grads.values()]
        self._net: torch.nn.Module = net
        self._best_model: torch.nn.Module = copy.deepcopy(net)
        self._device = next(net.parameters()).device
        self._sampled_clients: list[Client] = []
        self._sample_fn = random.choices if Server.SAMPLE_CLIENTS_WITH_REPLACEMENT else random.sample
        self._optimizer = torch.optim.SGD(self._net.parameters(),
                                          lr=Server.LEARNING_RATE,
                                          weight_decay=Server.WEIGHT_DECAY,
                                          momentum=Server.MOMENTUM)
        self._criterion = torch.nn.CrossEntropyLoss()
        self._val_loader: DataLoader = val_loader
        self._test_loader: DataLoader = test_loader
        self._aggregating_strategy = aggregating_strategy

    def federated_learn(self):

        pbar = tqdm(range(Server.NUM_ROUNDS))
        best_val_acc = 0.0
        best_round = 0
        for federated_round in pbar:
            # Train Round
            round_val_acc, round_val_loss = self.federated_round()

            # Best round so far?
            if round_val_acc > best_val_acc:
                logging.debug('Best val acc so far')
                best_round, best_val_acc = self.update_best_round_values(round_val_acc, federated_round)

            # progress bar and logging
            pbar.set_description(f'Round {federated_round} finished.'
                                 f' Acc {round_val_acc} '
                                 f'({best_val_acc} best acc till now,'
                                 f' best round {best_round}),'
                                 f' loss {round_val_loss}')
            if Config.LOG2WANDB:
                wandb.log({'val_acc': round_val_acc,
                           'val_loss': round_val_loss,
                           'best_epoch_validation_acc': best_val_acc,
                           'best_round': best_round})
        # Test
        acc, loss = self.test_net()
        logging.info(f'test loss {loss} acc {acc}')
        if Config.LOG2WANDB:
            wandb.log({'test_acc': acc, 'test_loss': loss})

    def update_best_round_values(self, current_val_acc, current_federated_round):
        best_round = current_federated_round
        best_val_acc = current_val_acc
        self._best_model = copy.copy(self._net)
        return best_round, best_val_acc

    def federated_round(self) -> tuple[float, float]:
        self.sample_clients()
        self.preform_train_round()
        self.aggregate_grads()
        self.update_net()
        return self.validate_net()

    def sample_clients(self):
        self._sampled_clients = self._sample_fn(self._train_clients, k=Server.NUM_CLIENT_AGG)
        logging.debug(f'sampled clients {str([c.cid for c in self._sampled_clients])}')

    def preform_train_round(self):
        assert self._sampled_clients, f'Expected sampled clients. Got {len(self._sampled_clients)} clients sampled'
        for c in self._sampled_clients:
            net_copy_for_client = copy.copy(self._net).to(self._device)
            zero_net_grads(net_copy_for_client)
            c.train(net=net_copy_for_client)

    def get_sampled_clients_grads(self) -> torch.Tensor:
        # collect private gradients
        layer_grad_batch_list = []
        for k in self._grads.keys():
            layer_grad_batch = torch.stack([c.grads[k] for c in self._sampled_clients])
            layer_grad_batch_list.append(layer_grad_batch)
        grad_batch_flattened = flatten_tensor(layer_grad_batch_list)
        return grad_batch_flattened

    def aggregate_grads(self):
        grad_batch_flattened: torch.Tensor = self.get_sampled_clients_grads()
        aggregated_flattened_grads: torch.Tensor = self._aggregating_strategy(grad_batch_flattened)
        del grad_batch_flattened
        self.store_aggregated_grads(aggregated_flattened_grads)

    def store_aggregated_grads(self, aggregated_grads_flattened: torch.Tensor):
        offset = 0
        for k in self._grads:
            num_elements: int = self._grads[k].numel()
            shape = self._grads[k].shape
            self._grads[k] += aggregated_grads_flattened[offset:offset + num_elements].reshape(shape)
            offset += num_elements

    def update_net(self):
        self._optimizer.zero_grad(set_to_none=False)
        for k, p in self._net.named_parameters():
            p.grad += self._grads[k]
            self._grads[k] = torch.zeros_like(p)
        self._optimizer.step()

    def validate_net(self) -> tuple[float, float]:
        return self.evaluate_net(clients=self._val_clients)

    def test_net(self) -> tuple[float, float]:
        return self.evaluate_net(clients=self._test_clients)

    @torch.no_grad()
    def evaluate_net(self, clients: list[Client]) -> tuple[float, float]:
        total_accuracy, total_loss = 0.0, 0.0
        for c in clients:
            acc, loss = c.evaluate(net=self._net)
            total_accuracy += (acc / len(self._test_clients))
            total_loss += (loss / len(self._test_clients))
        return total_accuracy, total_loss
