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
                 val_clients: list[Client],
                 test_clients: list[Client],
                 net: torch.nn.Module,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 aggregating_strategy: Callable[[torch.tensor], torch.tensor]):
        super().__init__(train_clients=private_clients,
                         val_clients=val_clients,
                         test_clients=test_clients,
                         net=net,
                         val_loader=val_loader,
                         test_loader=test_loader,
                         aggregating_strategy=aggregating_strategy)

        self._num_basis = GepServer.NUM_BASIS_ELEMENTS
        self._public_clients = public_clients
        self._pca: PCA = PCA()

    def _aggregate_grads(self):

        # train public clients to get new public gradients
        self._single_train_epoch_public_clients()

        # Update subspace basis according to public clients
        public_gradients = self._extract_public_gradients()
        self._compute_subspace(public_gradients)

        # get new grads
        grad_batch_flattened: torch.Tensor = self._get_sampled_clients_grads()

        # embed grads in subspace
        grad_batch_flattened_embedded: torch.Tensor = self._embed_grad(grad_batch_flattened)

        # aggregate grads
        aggregated_embedded_flattened_grads: torch.Tensor = self._aggregating_strategy(grad_batch_flattened_embedded)

        # reconstruct and reshape grads
        reconstructed_grads_flattened: torch.Tensor = self._project_back_embedding(aggregated_embedded_flattened_grads)
        self._store_aggregated_grads(reconstructed_grads_flattened)

        # delete temporary tensors
        del grad_batch_flattened, grad_batch_flattened_embedded,\
            aggregated_embedded_flattened_grads, reconstructed_grads_flattened

    def _single_train_epoch_public_clients(self):
        for public_client in self._public_clients:
            public_client.receive_net_from_server(net=self._net)
            public_client.train_single_epoch()

    def _compute_subspace(self, public_gradients: torch.Tensor):
        num_bases: int
        pub_error: float
        pca: PCA
        num_bases, pub_error, pca = get_bases(public_gradients, self._num_basis)
        self._pca = pca
        del public_gradients

    def _embed_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad_np: np.ndarray = grad.cpu().detach().numpy()
        embedding: np.ndarray = self._pca.transform(grad_np)
        return torch.from_numpy(embedding)

    def _project_back_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding_np: np.ndarray = embedding.cpu().detach().numpy()
        grad_np: np.ndarray = self._pca.inverse_transform(embedding_np)
        return torch.from_numpy(grad_np).to(self._device)

    def _extract_public_gradients(self):
        public_batch_grad_list = []
        for k in self._grads.keys():
            grad_batch = torch.stack([c.grads[k] for c in self._public_clients])
            public_batch_grad_list.append(grad_batch.reshape(grad_batch.shape[0], -1))
            del grad_batch
        public_gradients = flatten_tensor(public_batch_grad_list)
        return public_gradients
