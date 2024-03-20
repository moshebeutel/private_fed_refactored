import logging

import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from private_federated.models.utils import get_net_grads, zero_net_grads
from private_federated.train.utils import evaluate, clone_model, merge_model


class Client:
    INTERNAL_EPOCHS = 1
    CRITERION = CrossEntropyLoss()
    OPTIMIZER_TYPE = torch.optim.SGD
    OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-3, 'momentum': 0.9}
    PERSONALIZATION_WEIGHT = 0.1

    def __init__(self, cid: int, train_loader: DataLoader, eval_loader: DataLoader = None):
        self._id = cid
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._grads = {}
        self._net = None
        self._device = None

    def train(self):
        self._train(num_epochs=Client.INTERNAL_EPOCHS)

    def train_single_epoch(self):
        self._train(num_epochs=1)

    def _train(self, num_epochs: int = INTERNAL_EPOCHS):
        assert self._net is not None, 'Client must receive net must before '
        backup = clone_model(self._net).state_dict()
        self._net.train()

        criterion = Client.CRITERION
        optimizer = Client.OPTIMIZER_TYPE(params=self._net.parameters(), **Client.OPTIMIZER_PARAMS)

        for epoch in range(num_epochs):
            epoch_loss: float = 0.0
            for images, labels in self._train_loader:
                images, labels = images.to(self._device), labels.to(self._device)
                batch_size: int = len(labels)
                optimizer.zero_grad()

                outputs = self._net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                with torch.no_grad():
                    for i, p in self._net.named_parameters():
                        # self._grads[i] += (p.grad.data / batch_size)
                        self._grads[i] += (p.grad / 1e-3)

                optimizer.step()

                batch_loss: float = float(loss)

                del loss, images, labels
                epoch_loss += batch_loss

        with torch.no_grad():
            curr = self._net.state_dict()
            for k in curr:
                self._grads[k] = torch.clone(curr[k]) - torch.clone(backup[k])
        # acc, loss = evaluate(self._net, self._eval_loader, criterion)
        # logging.info(f'\ntrain loss: {loss:.4f}, train acc: {acc}')

        zero_net_grads(self._net)

    def evaluate(self) -> tuple[float, float]:
        return evaluate(net=self._net, loader=self._eval_loader, criterion=Client.CRITERION)

    def receive_net_from_server(self, net: torch.nn.Module):
        self._net = clone_model(net)
        self._new_net_updates()

    def merge_server_model_with_personal_model(self, net: torch):
        assert self._net is not None, 'Net not initialized. Nothing to merge to'
        self._net = merge_model(net, self._net, weight1=1 - Client.PERSONALIZATION_WEIGHT,
                                weight2=Client.PERSONALIZATION_WEIGHT)

        self._new_net_updates()

    @torch.no_grad()
    def _new_net_updates(self):
        zero_net_grads(self._net)
        self._grads = get_net_grads(self._net)
        self._device = next(self._net.parameters()).device

    @property
    def grads(self):
        return self._grads

    @property
    def cid(self):
        return self._id
