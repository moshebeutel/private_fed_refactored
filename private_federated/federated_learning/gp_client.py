import logging

import torch.nn
from torch.utils.data import DataLoader
from private_federated.data.dataset_factory import DatasetFactory
from private_federated.federated_learning.client import Client
from private_federated.models.pFedGP.utils import build_tree
from private_federated.models.utils import zero_net_grads
from private_federated.train.utils import clone_model


class GPClient(Client):

    def __init__(self, cid: int, train_loader: DataLoader, eval_loader: DataLoader = None):
        from private_federated.models.pFedGP.Learner import pFedGPFullLearner
        super().__init__(cid, train_loader, eval_loader)

        self._gp = pFedGPFullLearner(n_output=DatasetFactory.CLASSES_PER_USER)

    def _train(self, num_epochs: int = Client.INTERNAL_EPOCHS):
        assert self._net is not None, 'Client must receive net must before '
        backup = clone_model(self._net).state_dict()
        self._net.train()

        optimizer = Client.OPTIMIZER_TYPE(params=self._net.parameters(), **Client.OPTIMIZER_PARAMS)
        gp, label_map, _, __ = build_tree(gp=self._gp, net=self._net, loader=self._train_loader)
        gp.train()

        total_loss = 0

        for epoch in range(num_epochs):
            epoch_loss: float = 0.0
            epoch_size: int = 0
            optimizer.zero_grad()
            outputs_list, labels_list = [], []
            for images, labels in self._train_loader:
                images, labels = images.to(self._device), labels.to(self._device)
                batch_size: int = len(labels)
                epoch_size += batch_size

                outputs: torch.Tensor = self._net(images)
                outputs_list.append(outputs)
                labels_list.append(labels)

                del images, labels

            X: torch.Tensor = torch.cat(outputs_list, dim=0)
            Y: torch.Tensor = torch.cat(labels_list, dim=0)

            offset_labels = torch.tensor([label_map[lbl.item()] for lbl in Y],
                                         dtype=Y.dtype,
                                         device=Y.device)
            loss = gp(X, offset_labels)
            assert hasattr(loss, 'backward'), ("Expected loss function"
                                               " that can propagate gradients "
                                               "backward")
            loss.backward()

            optimizer.step()
            epoch_loss = float(loss) / epoch_size
            total_loss += (epoch_loss / num_epochs)
            del loss, offset_labels, X, Y, outputs_list, labels_list

        with torch.no_grad():
            curr = self._net.state_dict()
            grads_amp = 0
            for k in curr:
                self._grads[k] = torch.clone(curr[k]) - torch.clone(backup[k])
                grads_amp = max(grads_amp,
                                float(torch.max(torch.abs(self._grads[k]))))

        del gp.tree

        logging.debug(f'Client {self.cid} train loss: {total_loss:.4f} Grads amplitude: {grads_amp:.4f}')

        acc, loss = self.evaluate()

        logging.debug(f'Client {self.cid} eval loss: {loss:.4f} acc {acc:.4f}')

        zero_net_grads(self._net)
        return total_loss, acc, loss

    @torch.no_grad()
    def evaluate(self):
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
