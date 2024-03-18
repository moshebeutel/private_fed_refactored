import logging
import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from private_federated.models.utils import get_net_grads, zero_net_grads
from private_federated.train.utils import evaluate, clone_model, merge_model


class Client:
    INTERNAL_EPOCHS = 5
    CRITERION = CrossEntropyLoss()
    OPTIMIZER_TYPE = torch.optim.SGD
    OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-3, 'momentum': 0.9}
    PESONALIZATION_WEIGHT = 0.1

    def __init__(self, cid: int, loader: DataLoader):
        self._id = cid
        self._loader = loader
        self._grads = {}
        self._net = None
        self._sample_counter = 0

    def train(self, net: torch.nn.Module):
        self._sample_counter += 1
        device = next(net.parameters()).device
        criterion = Client.CRITERION
        net4train: torch.nn.Module = clone_model(net)
        net4train.train()
        zero_net_grads(net4train)
        self._grads = get_net_grads(net4train)
        optimizer = Client.OPTIMIZER_TYPE(params=net4train.parameters(), **Client.OPTIMIZER_PARAMS)
        for epoch in range(Client.INTERNAL_EPOCHS):
            epoch_loss: float = 0.0
            epoch_size: int = 0
            for images, labels in self._loader:
                images, labels = images.to(device), labels.to(device)
                batch_size: int = len(labels)
                optimizer.zero_grad()

                outputs = net4train(images)
                loss = criterion(outputs, labels)
                loss.backward()

                for i, p in net4train.named_parameters():
                    self._grads[i] += (p.grad.data / batch_size)

                optimizer.step()

                batch_loss: float = float(loss)

                epoch_size += batch_size
                del loss, images, labels
                epoch_loss += batch_loss

        self._net = clone_model(net4train)

        with torch.no_grad():
            for i in self._grads:
                self._grads[i] /= Client.INTERNAL_EPOCHS

    def evaluate(self, net: torch.nn.Module, fine_tune: bool = False, local_weight: float = 0.0) -> tuple[float, float]:

        self._net = clone_model(net) if (self._net is None or local_weight == 0.0) else \
            merge_model(net, self._net, include_grads=False, weight1=1 - local_weight, weight2=local_weight)
        if fine_tune:
            self.train(self._net)
        return evaluate(net=self._net, loader=self._loader, criterion=Client.CRITERION)

    @property
    def grads(self):
        return self._grads

    @property
    def cid(self):
        return self._id
