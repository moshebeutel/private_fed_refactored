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

    @torch.no_grad()
    def update_net(self, net: torch.nn.Module):
        if self._net is None:
            self._net = clone_model(net)
            zero_net_grads(self._net)
            return
        else:
            w1 = Client.PESONALIZATION_WEIGHT
            w2 = 1 - Client.PESONALIZATION_WEIGHT
            if self._sample_counter > 2:
                logging.info(f"Client {self.cid} sampled {self._sample_counter}")
            updated = False
            # for local_param, update_param in zip(self._net.parameters(), net.parameters()):
            #     assert local_param.shape == update_param.shape, \
            #         f"{local_param.shape} vs {update_param.shape} is expected to have the same shape"
            #     if torch.allclose(update_param, local_param):
            #
            #         # logging.debug(f"ALL CLOSE CLIENT {self.cid}")
            #         pass
            #     else:
            #         logging.error(f"UPDATING params for {self.cid}")
            #         # local_param.copy_(w1 * local_param + w2 * update_param)
            #         updated = True
            local_state = self._net.state_dict()
            update_state = net.state_dict()
            for k, p in self._net.named_parameters():

                if torch.allclose(p, update_state[k]):

                    # logging.debug(f"ALL CLOSE CLIENT {self.cid}")
                    pass
                else:
                    logging.error(f"UPDATING params for {self.cid}")
                    # local_param.copy_(w1 * local_param + w2 * update_param)
                    updated = True

            new_state = {}
            for k, v in local_state.items():
                new_state[k] = w1 * v + w2 * update_state[k]
            self._net.load_state_dict(new_state)

            for k, p in self._net.named_parameters():

                if torch.allclose(p, update_state[k]):

                    # logging.debug(f"ALL CLOSE CLIENT {self.cid}")
                    pass
                else:
                    logging.error(f"UPDATING params for {self.cid}")
                    # local_param.copy_(w1 * local_param + w2 * update_param)
                    updated = True
            if self._sample_counter > 2:
                logging.info(f"Client {self.cid} sampled {self._sample_counter} updated {updated}")

            self._net.load_state_dict(net.state_dict())

    def train(self, net: torch.nn.Module):
        self._sample_counter += 1
        device = next(net.parameters()).device
        criterion = Client.CRITERION

        # net4train: torch.nn.Module = clone_model(net) if self._net is None else\
        #     merge_model(model1=net, model2=self._net, include_grads=False, weight1=0.9, weight2=0.1)

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

    @torch.no_grad()
    def evaluate(self, net: torch.nn.Module, local_weight: float = 0.0) -> tuple[float, float]:
        # self.update_net(net)
        # net = copy.deepcopy(net)
        # return evaluate(net=net, loader=self._loader, criterion=Client.CRITERION)
        self._net = clone_model(net) if (self._net is None or local_weight == 0.0) else\
            merge_model(net, self._net, include_grads=False, weight1=1-local_weight, weight2=local_weight)
        return evaluate(net=self._net, loader=self._loader, criterion=Client.CRITERION)

    @property
    def grads(self):
        return self._grads

    @property
    def cid(self):
        return self._id
