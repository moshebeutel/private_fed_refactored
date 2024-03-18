import torch.nn
from torch.utils.data import DataLoader
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.federated_learning.client import Client
from private_federated.models.pFedGP.utils import build_tree


class GPClient(Client):

    def __init__(self, cid: int, train_loader: DataLoader, eval_loader: DataLoader = None):
        from private_federated.models.pFedGP.Learner import pFedGPFullLearner
        super().__init__(cid, train_loader, eval_loader)

        self._gp = pFedGPFullLearner(n_output=DataLoadersGenerator.CLASSES_PER_USER)

    def _train(self, num_epochs: int = Client.INTERNAL_EPOCHS):
        assert self._net is not None, 'Client must receive net must before '
        self._net.train()

        optimizer = Client.OPTIMIZER_TYPE(params=self._net.parameters(), **Client.OPTIMIZER_PARAMS)
        gp, label_map, _, __ = build_tree(gp=self._gp, net=self._net, loader=self._train_loader)
        gp.train()

        for epoch in range(num_epochs):
            epoch_loss: float = 0.0
            optimizer.zero_grad()
            is_first_iter = True
            for images, labels in self._train_loader:
                images, labels = images.to(self._device), labels.to(self._device)

                outputs: torch.Tensor = self._net(images)

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

            epoch_size: float = float(len(Y))

            with torch.no_grad():
                for i, p in self._net.named_parameters():
                    self._grads[i] += (p.grad.data / epoch_size)

            optimizer.step()
            epoch_loss += float(loss)
            del loss, offset_labels

        with torch.no_grad():
            for i, p in self._net.named_parameters():
                self._grads[i] /= float(num_epochs)

        del gp.tree

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        eval_accuracy, total, eval_loss = 0.0, 0.0, 0.0

        gp, label_map, X_train, Y_train = build_tree(gp=self._gp, net=self._net, loader=self._train_loader)
        gp.eval()
        is_first_iter = True
        with torch.no_grad():
            for data in self._eval_loader:
                images, labels = data[0].to(self._device), data[1].to(self._device)

                Y_test = torch.tensor([label_map[lbl.item()] for lbl in labels], dtype=labels.dtype,
                                      device=labels.device)

                X_test = self._net(images)
                loss, pred = gp.forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)
                is_first_iter = False

                eval_loss += float(loss)
                predicted = pred.argmax(1)
                total += Y_test.size(0)
                eval_accuracy += (predicted == Y_test).sum().item()
                del predicted, loss, images, labels, pred, Y_test, X_test
        del X_train, Y_train
        self._gp = gp
        eval_accuracy /= float(total)
        eval_loss /= float(total)
        return eval_accuracy, eval_loss
