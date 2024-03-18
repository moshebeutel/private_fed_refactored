import logging
import random
from typing import Callable
import torch.nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from private_federated.common.config import Config
from private_federated.differential_privacy.gep.utils import flatten_tensor
from private_federated.federated_learning.client import Client
from private_federated.train.utils import evaluate, clone_model
from private_federated.models.utils import get_net_grads, zero_net_grads


class Server:
    NUM_ROUNDS = 80
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
        self._net: torch.nn.Module = clone_model(net)
        zero_net_grads(self._net)
        self._grads: {str: torch.tensor} = get_net_grads(self._net)
        self._shapes = [g.shape for g in self._grads.values()]
        self._best_model: torch.nn.Module = clone_model(net)
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
        # Test the best model achieved
        self._net = clone_model(self._best_model).to(self._device)

        acc, loss = self._test_net()
        logging.info(f'test loss {loss} acc {acc}')
        if Config.LOG2WANDB:
            wandb.log({'test_acc': acc, 'test_loss': loss})

        acc, loss = self._validate_train_clients_test_set()
        logging.info(f'train clients test loss {loss} acc {acc}')
        if Config.LOG2WANDB:
            wandb.log({'test_acc_train_clients': acc, 'test_loss_train_clients': loss})

        acc, loss = self._test_with_finetune()
        logging.info(f'finetune test loss {loss} acc {acc}')
        if Config.LOG2WANDB:
            wandb.log({'test_acc_with_finetune': acc, 'test_loss_with_finetune': loss})

    def update_best_round_values(self, current_val_acc, current_federated_round):
        best_round = current_federated_round
        best_val_acc = current_val_acc
        self._best_model = clone_model(self._net)
        return best_round, best_val_acc

    def federated_round(self) -> tuple[float, float]:
        self.sample_clients()
        self.preform_train_round()
        self.aggregate_grads()
        self._update_net()
        return self._validate_net()

    def sample_clients(self):
        self._sampled_clients = self._sample_fn(self._train_clients, k=Server.NUM_CLIENT_AGG)
        logging.debug(f'\nsampled clients {str([c.cid for c in self._sampled_clients])}')

    def preform_train_round(self):
        assert self._sampled_clients, f'Expected sampled clients. Got {len(self._sampled_clients)} clients sampled'
        for c in self._sampled_clients:
            c.train(net=self._net)

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

    def _update_net(self):
        self._optimizer.zero_grad(set_to_none=False)
        for k, p in self._net.named_parameters():
            p.grad += self._grads[k]
            self._grads[k] = torch.zeros_like(p)
        self._optimizer.step()

    def _validate_net(self) -> tuple[float, float]:
        return self._evaluate_net(clients=self._val_clients)

    def _test_net(self) -> tuple[float, float]:
        return self._evaluate_net(clients=self._test_clients)

    def _evaluate_net(self, clients: list[Client], fine_tune: bool = False, local_weight: float = 0.0) -> tuple[float, float]:
        total_accuracy, total_loss = 0.0, 0.0
        for c in clients:
            acc, loss = c.evaluate(net=self._net, fine_tune=fine_tune, local_weight=local_weight)
            total_accuracy += acc
            total_loss += loss
        return total_accuracy/float(len(clients)), total_loss/float(len(clients))

    def _validate_train_clients_test_set(self):
        return self._evaluate_net(clients=self._train_clients, local_weight=Client.PESONALIZATION_WEIGHT)

    def _test_with_finetune(self):
        return self._evaluate_net(clients=self._test_clients, fine_tune=True, local_weight=Client.PESONALIZATION_WEIGHT)
