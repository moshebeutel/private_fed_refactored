from typing import Callable
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from private_federated.differential_privacy.gep.utils import flatten_tensor, get_bases
from private_federated.federated_learning.client import Client
from private_federated.federated_learning.server import Server


class GepServer(Server):
    NUM_BASIS_ELEMENTS = 10

    def __init__(self,
                 public_clients: list[Client],
                 private_clients: list[Client],
                 net: torch.nn.Module,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 aggregating_strategy: Callable[[torch.tensor], torch.tensor]):
        super().__init__(clients=private_clients,
                         net=net,
                         val_loader=val_loader,
                         test_loader=test_loader,
                         aggregating_strategy=aggregating_strategy)

        self._num_basis = GepServer.NUM_BASIS_ELEMENTS
        self._public_clients = public_clients
        self._pca: PCA = PCA()

    def sample_clients(self):
        super().sample_clients()
        self._sampled_clients = self._public_clients + self._sampled_clients

    def aggregate_grads(self):

        # Update subspace basis according to public clients
        public_gradients = self.extract_public_gradients()
        self.compute_subspace(public_gradients)

        # remove public clients - if they do not contribute to learning
        self._sampled_clients = self._sampled_clients[len(self._public_clients):]

        # get new grads
        grad_batch: torch.Tensor = self.get_sampled_clients_grads()
        grad_batch_embedding: torch.Tensor = self.embed_grad(grad_batch)
        aggregated_embedded_grads_flattened: torch.Tensor = self._aggregating_strategy(grad_batch_embedding)
        aggregated_grads_flattened: torch.Tensor = self.project_back_embedding(aggregated_embedded_grads_flattened)
        self.store_aggregated_grads(aggregated_grads_flattened)
        del grad_batch, grad_batch_embedding

    def compute_subspace(self, public_gradients: torch.Tensor):
        num_bases: int
        pub_error: float
        pca: PCA
        num_bases, pub_error, pca = get_bases(public_gradients, self._num_basis)
        self._pca = pca
        del public_gradients

    def embed_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad_np: np.ndarray = grad.cpu().detach().numpy()
        embedding: np.ndarray = self._pca.transform(grad_np)
        return torch.from_numpy(embedding)

    def project_back_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding_np: np.ndarray = embedding.cpu().detach().numpy()
        grad_np: np.ndarray = self._pca.inverse_transform(embedding_np)
        return torch.from_numpy(grad_np).to(self._device)

    def extract_public_gradients(self):
        public_batch_grad_list = []
        for k in self._grads.keys():
            grad_batch = torch.stack([c.grads[k] for c in self._public_clients])
            public_batch_grad_list.append(grad_batch.reshape(grad_batch.shape[0], -1))
            del grad_batch
        public_gradients = flatten_tensor(public_batch_grad_list)
        return public_gradients
