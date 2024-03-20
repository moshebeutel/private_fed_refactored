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
from private_federated.federated_learning.utils import evaluate_clients
from private_federated.models.utils import get_net_grads, zero_net_grads
from private_federated.train.utils import clone_model, merge_model, evaluate


class Server:
    NUM_ROUNDS = 30
    NUM_CLIENT_AGG: int = 5
    SAMPLE_CLIENTS_WITH_REPLACEMENT: bool = False
    LEARNING_RATE: float = 1.0
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
        self._sampled_clients_history: list[Client] = []
        self._sample_fn = random.choices if Server.SAMPLE_CLIENTS_WITH_REPLACEMENT else random.sample
        self._optimizer = torch.optim.SGD(self._net.parameters(),
                                          lr=Server.LEARNING_RATE,
                                          weight_decay=Server.WEIGHT_DECAY,
                                          momentum=Server.MOMENTUM)
        # self._criterion = torch.nn.CrossEntropyLoss()
        self._val_loader: DataLoader = val_loader
        self._test_loader: DataLoader = test_loader
        self._aggregating_strategy = aggregating_strategy
        self._last_val_acc: float = 0.0
        self._best_val_acc: float = 0.0
        self._best_round: int = 0
        self._progress_bar = tqdm(range(Server.NUM_ROUNDS))

    def federated_learn(self):
        """
        Federated learning server entry point
        """
        # main federated learning loop
        for federated_train_round in self._progress_bar:
            self._federated_round()

            self._check_best_round(federated_train_round)

            self._update_progress_bar_desc(federated_train_round)

        self._test_net()

    def _update_progress_bar_desc(self, federated_train_round):
        self._progress_bar.set_description(f'Round {federated_train_round} finished.'
                                           f' Acc {self._last_val_acc} '
                                           f'({self._best_val_acc} best acc till now,'
                                           f' best round {self._best_round})')

    def _federated_round(self):
        """
        Federated learning single round. Contains calls to subprocess subroutines:
        1. client sampling,
        2. launch client local train using current network state,
        3. aggregate gradients from the sampled clients
        4. update the weights of the network on the server
        4. evaluate the model on the server
        """
        self._sample_clients()
        self._preform_train_round(self._sampled_clients)
        self._aggregate_grads()
        self._update_server_model()
        self._evaluate_server_model()

    def _sample_clients(self):
        """
        Sample clients according to configured number of clients to aggregate and with or without replacement.
        Maintain a list of already sampled clients
        """
        self._sampled_clients = self._sample_fn(self._train_clients, k=Server.NUM_CLIENT_AGG)
        self._sampled_clients_history.extend(self._sampled_clients)
        self._sampled_clients_history = list(set(self._sampled_clients_history))
        logging.debug(f'\nsampled clients {str([c.cid for c in self._sampled_clients])}')

    def _preform_train_round(self, clients: list[Client]):
        """
        Train round: send sampled clients the current state of the network and launch a local train on each client
        :param clients (list[Client]): list of clients participating in the train round
        """
        assert clients, f'Expected clients list. Got {len(clients)} clients'
        for c in clients:
            c.receive_net_from_server(net=self._net)
            c.train()

    def _get_clients_grads(self, clients: list[Client]) -> torch.Tensor:
        """
        Get gradients generated at last local train of each client in the given clients list\
        :param clients (list[Client]): list of clients requested to hand over their gradients
        :return (torch.Tensor): A tensor of concatenated flattened per client gradients
        """
        layer_grad_batch_list = []
        for k in self._grads.keys():
            layer_grad_batch = torch.stack([torch.clone(c.grads[k]) for c in clients])
            layer_grad_batch_list.append(layer_grad_batch)
        grad_batch_flattened = flatten_tensor(layer_grad_batch_list)
        return grad_batch_flattened

    def _get_sampled_clients_grads(self) -> torch.Tensor:
        """
        collect private gradients
        :return (torch.Tensor): A tensor of concatenated flattened per sampled private client gradients
        """
        return self._get_clients_grads(self._sampled_clients)

    def _aggregate_grads(self):
        """
        Aggregate private gradients over sampled clients.
        Use an aggregation strategy. Store the aggregated result.
        """
        grad_batch_flattened: torch.Tensor = self._get_sampled_clients_grads()
        assert torch.linalg.norm(grad_batch_flattened) > 0.0, f'Gradient batch norm is zero'
        aggregated_flattened_grads: torch.Tensor = self._aggregating_strategy(grad_batch_flattened)
        assert torch.linalg.norm(aggregated_flattened_grads) > 0.0, f'Gradient aggregated norm is zero'
        # del grad_batch_flattened
        self._store_aggregated_grads(aggregated_flattened_grads)

    def _store_aggregated_grads(self, aggregated_grads_flattened: torch.Tensor):
        """
        Store aggregated gradients in a local state dict.
        Unflatten the tensor to the original network parameters shapes
        :param aggregated_grads_flattened (torch.Tensor): The aggregated grads.
                                                          The grads are flattened to a single row tensor
        """
        offset = 0
        for k in self._grads:
            num_elements: int = self._grads[k].numel()
            shape = self._grads[k].shape
            grad_update = aggregated_grads_flattened[offset:offset + num_elements].reshape(shape)
            assert torch.linalg.norm(grad_update) > 0.0, f'The gradient update of {k} is zero norm'
            logging.debug(f'before store       {torch.linalg.norm(self._grads[k])}')
            self._grads[k] += aggregated_grads_flattened[offset:offset + num_elements].reshape(shape)
            logging.debug(f'after store       {torch.linalg.norm(self._grads[k])}')
            offset += num_elements

    def _update_server_model(self):
        """
        Update the server model using the local grads state dict
        """

        grads_model = clone_model(self._net)
        grads_model.load_state_dict(self._grads)
        self._net = merge_model(model1=self._net, model2=grads_model,
                                weight1=1-Server.LEARNING_RATE, weight2=Server.LEARNING_RATE)

    def _evaluate_server_model(self):
        """
        Evaluate the server model
        """

        self._last_val_acc, val_loss = self._evaluate_train_clients_on_their_test_set()
        # self._last_val_acc, val_loss = evaluate(net=self._net, loader=self._val_loader, criterion=Client.CRITERION)
        logging.info(f'\nvalidation accuracy: {self._last_val_acc}')
        if Config.LOG2WANDB:
            wandb.log({'val_acc': self._last_val_acc,
                       'val_loss': val_loss})

    def _test_net(self):

        # Use the best model achieved for test
        self._net = clone_model(self._best_model).to(self._device)
        # self._last_val_acc, val_loss = evaluate(net=self._net, loader=self._val_loader, criterion=Client.CRITERION)

        acc, loss = self._evaluate_test_clients()
        logging.info(f'test loss {loss} acc {acc}')
        if Config.LOG2WANDB:
            wandb.log({'test_acc': acc, 'test_loss': loss})

        acc, loss = self._evaluate_train_clients_on_their_test_set_with_personalization()
        logging.info(f'train clients test loss {loss} acc {acc}')
        if Config.LOG2WANDB:
            wandb.log({'test_acc_train_clients': acc, 'test_loss_train_clients': loss})

        acc, loss = self._evaluate_test_clients_with_finetune()
        logging.info(f'finetune test loss {loss} acc {acc}')
        if Config.LOG2WANDB:
            wandb.log({'test_acc_with_finetune': acc, 'test_loss_with_finetune': loss})

    def _evaluate_val_clients(self) -> tuple[float, float]:
        for client in self._val_clients:
            client.receive_net_from_server(net=self._net)
        return evaluate_clients(clients=self._val_clients)

    def _evaluate_val_clients_with_finetune(self) -> tuple[float, float]:
        # fine tune model for each client
        for client in self._val_clients:
            client.train_single_epoch()
            client.merge_server_model_with_personal_model(net=self._net)

        return evaluate_clients(clients=self._val_clients)

    def _evaluate_test_clients(self) -> tuple[float, float]:
        for client in self._test_clients:
            client.receive_net_from_server(net=self._net)
        return evaluate_clients(clients=self._test_clients)

    def _evaluate_test_clients_with_finetune(self) -> tuple[float, float]:
        # fine tune model for each client
        for client in self._test_clients:
            client.train_single_epoch()
            client.merge_server_model_with_personal_model(net=self._net)

        return evaluate_clients(clients=self._test_clients)

    def _evaluate_train_clients_on_their_test_set(self) -> tuple[float, float]:
        for client in self._sampled_clients_history:
            client.receive_net_from_server(net=self._net)
        return evaluate_clients(clients=self._sampled_clients_history)

    def _evaluate_train_clients_on_their_test_set_with_personalization(self) -> tuple[float, float]:
        # each client blend server weights with his local
        for client in self._sampled_clients_history:
            client.merge_server_model_with_personal_model(net=self._net)

        return evaluate_clients(clients=self._sampled_clients_history)

    def _check_best_round(self, federated_train_round: int) -> None:
        if self._last_val_acc > self._best_val_acc:
            logging.debug(f'Best val acc so far {self._last_val_acc} > {self._best_val_acc}')
            self._update_best_round_values(federated_train_round)

    def _update_best_round_values(self, current_federated_round) -> None:
        self._best_round = current_federated_round
        self._best_val_acc = self._last_val_acc
        self._best_model = clone_model(self._net)
        if Config.LOG2WANDB:
            wandb.log({'best_epoch_validation_acc': self._best_val_acc,
                       'best_round': self._best_round})
