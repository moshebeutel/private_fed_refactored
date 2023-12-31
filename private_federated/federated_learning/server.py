import random
from copy import copy
from typing import Callable
import torch.nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from private_federated.federated_learning.client import Client
from private_federated.train.utils import init_net_grads, get_net_grads, evaluate


class Server:
    NUM_ROUNDS = 100
    NUM_CLIENT_AGG: int = 50
    SAMPLE_CLIENTS_WITH_REPLACEMENT: bool = True
    LEARNING_RATE: float = 0.0001
    WEIGHT_DECAY: float = 1e-3
    MOMENTUM: float = 0.9

    def __init__(self, clients: list[Client], net: torch.nn.Module, val_loader: DataLoader, test_loader: DataLoader,
                 aggregating_strategy: Callable[[torch.tensor], torch.tensor]):
        self._clients: list[Client] = clients
        init_net_grads(net)
        self._grads: {str: torch.tensor} = get_net_grads(net)
        self._net: torch.nn.Module = net
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
        wandb.init(project="emg_gp_moshe", entity="emg_diff_priv", name='federated simple')

        pbar = tqdm(range(Server.NUM_ROUNDS))
        best_val_acc = 0.0
        best_round = 0
        for i in pbar:
            acc, loss = self.federated_round()
            best_round = i if best_val_acc < acc else best_round
            best_val_acc = max(best_val_acc, acc)
            pbar.set_description(f'Round {i} finished. Acc {acc} ({best_val_acc} best acc till now,'
                                 f' best round {best_round}), loss {loss}')
            wandb.log({'val_acc': acc, 'val_loss': loss, 'best_epoch_validation_acc': best_val_acc,
                       'best_round': best_round})

        acc, loss = self.eval_net(loader=self._test_loader)
        print(f'test loss {loss} acc {acc}')
        wandb.log({'test_acc': acc, 'test_loss': loss, 'best_epoch_validation_acc': best_val_acc})

    def federated_round(self):
        self.sample_clients()
        self.preform_train_round()
        self.aggregate_grads()
        self.update_net()
        return self.eval_net(loader=self._val_loader)

    def sample_clients(self):
        self._sampled_clients = self._sample_fn(self._clients, k=Server.NUM_CLIENT_AGG)
        # print([c.cid for c in self._sampled_clients])

    def preform_train_round(self):
        assert self._sampled_clients, f'Expected sampled clients. Got {len(self._sampled_clients)} clients sampled'
        for c in self._sampled_clients:
            net_copy_for_client = copy(self._net).to(self._device)
            c.train(net=net_copy_for_client)

    def aggregate_grads(self):
        for k in self._grads.keys():
            grad_batch = torch.stack([c.grads[k] for c in self._sampled_clients])
            self._grads[k] = self._aggregating_strategy(grad_batch)
            del grad_batch

    def update_net(self):
        self._optimizer.zero_grad(set_to_none=False)
        for k, p in self._net.named_parameters():
            p.grad += self._grads[k]
            self._grads[k] = torch.zeros_like(p)
        self._optimizer.step()

    def eval_net(self, loader: DataLoader):
        return evaluate(net=self._net, loader=loader, criterion=self._criterion)
