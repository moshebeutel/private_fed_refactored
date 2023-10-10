import torch.nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from private_federated.train.utils import get_net_grads


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
                optimizer.zero_grad()

                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                for i, p in net.named_parameters():
                    self._grads[i] += p.grad.data

                optimizer.step()

                batch_loss: float = float(loss)
                batch_size: int = len(labels)
                epoch_size += batch_size
                del loss, images, labels
                epoch_loss += batch_loss
            #     print(f'    client {self._id} batch loss {batch_loss/batch_size}')
            # print(f'client {self._id} epoch {epoch} epoch loss {epoch_loss/epoch_size}')

    @property
    def grads(self):
        return self._grads

    @property
    def cid(self):
        return self._id
