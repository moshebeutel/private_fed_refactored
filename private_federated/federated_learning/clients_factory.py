import torch

from private_federated.federated_learning.client import Client


class ClientFactory:
    def __init__(self, loaders: dict[int:dict[str:torch.utils.data.DataLoader]]):
        self._clients = [Client(cid=cid, loader=loaders[cid]['train']) for cid in loaders]

    @property
    def clients(self):
        return self._clients
