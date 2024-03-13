import torch.nn
from torch.utils.data import DataLoader
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.federated_learning.client import Client
from private_federated.models.pFedGP.Learner import pFedGPFullLearner
from private_federated.models.pFedGP.utils import build_tree
from private_federated.models.utils import get_net_grads


class GPClient(Client):

    def __init__(self, cid: int, loader: DataLoader, eval_loader: DataLoader = None):
        super().__init__(cid, loader)

        self._eval_loader = eval_loader
        self._gp = pFedGPFullLearner(n_output=DataLoadersGenerator.CLASSES_PER_USER)

    def train(self, net: torch.nn.Module):
        device = next(net.parameters()).device
        net.train()
        self._grads = get_net_grads(net)
        optimizer = Client.OPTIMIZER_TYPE(params=net.parameters(), **Client.OPTIMIZER_PARAMS)
        gp, label_map, _, __ = build_tree(gp=self._gp, net=net, loader=self._loader)
        gp.train()

        for epoch in range(Client.INTERNAL_EPOCHS):
            epoch_loss: float = 0.0
            optimizer.zero_grad()
            is_first_iter = True
            for images, labels in self._loader:
                images, labels = images.to(device), labels.to(device)

                outputs: torch.Tensor = net(images)

                X: torch.Tensor
                Y: torch.Tensor
                X = torch.cat((X, outputs), dim=0) if not is_first_iter else outputs
                Y = torch.cat((Y, labels), dim=0) if not is_first_iter else labels
                is_first_iter = False
                del images, labels

            offset_labels = torch.tensor([label_map[lbl.item()] for lbl in Y], dtype=Y.dtype,
                                         device=Y.device)
            loss = gp(X, offset_labels)
            assert hasattr(loss, 'backward'), ("Expected loss function that can propagate gradients "
                                               "backward")
            loss.backward()

            epoch_size = float(len(Y))

            with torch.no_grad():
                for i, p in net.named_parameters():
                    self._grads[i] += (p.grad.data / epoch_size)

            optimizer.step()
            epoch_loss += float(loss)

        with torch.no_grad():
            for i, p in net.named_parameters():
                self._grads[i] /= Client.INTERNAL_EPOCHS

        gp.tree = None

    def evaluate(self, net: torch.nn.Module) -> tuple[float, float]:
        eval_accuracy, total, eval_loss = 0.0, 0.0, 0.0
        device = next(net.parameters()).device
        net.eval()
        gp, label_map, X_train, Y_train = build_tree(gp=self._gp, net=net, loader=self._loader)
        gp.eval()
        is_first_iter = True
        with torch.no_grad():
            for data in self._eval_loader:
                images, labels = data[0].to(device), data[1].to(device)

                Y_test = torch.tensor([label_map[lbl.item()] for lbl in labels], dtype=labels.dtype,
                                      device=labels.device)

                X_test = net(images)
                loss, pred = gp.forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)
                is_first_iter = False

                eval_loss += float(loss)
                predicted = pred.argmax(1)
                total += Y_test.size(0)
                eval_accuracy += (predicted == Y_test).sum().item()
                del predicted, loss, images, labels, pred, Y_test, X_test
        del X_train, Y_train
        self._gp = gp
        eval_accuracy /= total
        eval_loss /= total
        return eval_accuracy, eval_loss

