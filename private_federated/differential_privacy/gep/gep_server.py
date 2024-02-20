from typing import Callable
import torch
from torch.utils.data import DataLoader
from differential_privacy.gep.gep import GEP
from differential_privacy.gep.utils import flatten_tensor
from federated_learning.client import Client
from federated_learning.server import Server


class GepServer(Server):
    def __init__(self,
                 public_clients: list[Client],
                 private_clients: list[Client],
                 net: torch.nn.Module,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 aggregating_strategy: Callable[[torch.tensor], torch.tensor]):
        super().__init__(clients=private_clients, net=net, val_loader=val_loader, test_loader=test_loader,
                         aggregating_strategy=aggregating_strategy)

        self._gep = GEP(public_clients=public_clients, num_bases=len(public_clients), batch_size=Server.NUM_CLIENT_AGG)
        self._public_clients = public_clients

    def sample_clients(self):
        super().sample_clients()
        super()._sampled_clients = self._public_clients + super()._sampled_clients

    def aggregate_grads(self):
        public_batch_grad_list = []
        for k in self._grads.keys():
            grad_batch = torch.stack([c.grads[k] for c in self._public_clients])
            public_batch_grad_list.append(grad_batch.reshape(grad_batch.shape[0], -1))
            del grad_batch
        anchor_gradients = flatten_tensor(public_batch_grad_list)
        self._gep.get_anchor_space(anchor_gradients)
        del anchor_gradients
        # remove public clients - if they do not contribute to learning
        # and then collect private gradients
        super()._sampled_clients = super()._sampled_clients[len(self._public_clients):]
        super().aggregate_grads()
