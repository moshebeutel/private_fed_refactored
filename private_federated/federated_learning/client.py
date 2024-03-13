import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from private_federated.models.utils import get_net_grads
from private_federated.train.utils import evaluate


class Client:
    INTERNAL_EPOCHS = 5
    CRITERION = CrossEntropyLoss()
    OPTIMIZER_TYPE = torch.optim.SGD
    OPTIMIZER_PARAMS = {'lr': 1e-3, 'weight_decay': 1e-3, 'momentum': 0.9}

    def __init__(self, cid: int, loader: DataLoader):
        self._id = cid
        self._loader = loader
        self._grads = {}

    def train(self, net: torch.nn.Module):
        device = next(net.parameters()).device
        criterion = Client.CRITERION
        net.train()
        self._grads = get_net_grads(net)
        optimizer = Client.OPTIMIZER_TYPE(params=net.parameters(), **Client.OPTIMIZER_PARAMS)
        for epoch in range(Client.INTERNAL_EPOCHS):
            epoch_loss: float = 0.0
            epoch_size: int = 0
            for images, labels in self._loader:
                images, labels = images.to(device), labels.to(device)
                batch_size: int = len(labels)
                optimizer.zero_grad()

                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                for i, p in net.named_parameters():
                    self._grads[i] += (p.grad.data / batch_size)

                optimizer.step()

                batch_loss: float = float(loss)

                epoch_size += batch_size
                del loss, images, labels
                epoch_loss += batch_loss

        with torch.no_grad():
            for i, p in net.named_parameters():
                self._grads[i] /= Client.INTERNAL_EPOCHS

    def evaluate(self, net: torch.nn.Module) -> tuple[float, float]:
        return evaluate(net=net, loader=self._loader, criterion=Client.CRITERION)

    @property
    def grads(self):
        return self._grads

    @property
    def cid(self):
        return self._id
