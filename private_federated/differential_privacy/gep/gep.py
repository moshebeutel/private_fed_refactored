from copy import copy

import numpy as np
from torch import nn
import torch
from differential_privacy.gep.utils import flatten_tensor, get_bases, check_approx_error, get_approx_grad, clip_column


class GEP(nn.Module):

    def __init__(self, public_clients, num_bases, batch_size, clip0=1, clip1=1):
        super(GEP, self).__init__()
        self._public_clients = public_clients
        self.num_bases_list = []
        self.num_bases = num_bases
        self.clip0 = clip0
        self.clip1 = clip1
        self.batch_size = batch_size
        self.approx_error_pca_private = {}
        self._selected_pca_list: list[torch.tensor] = []
        self._approx_errors_pca_public = []

    @property
    def approx_errors_pca_public(self):
        return self._approx_errors_pca_public

    @property
    def selected_bases_list(self):
        return self._selected_pca_list

    def get_anchor_gradients(self, net):
        cur_batch_grad_list = []
        for p in net.parameters():
            if hasattr(p, 'grad_batch'):
                grad_batch = p.grad_batch[:len(self.public_users)]
                cur_batch_grad_list.append(grad_batch.reshape(grad_batch.shape[0], -1))
        return flatten_tensor(cur_batch_grad_list)

    def get_anchor_space(self, net):
        anchor_grads = self.get_anchor_gradients(net)
        with (torch.no_grad()):
            num_param_list = self.num_param_list
            selected_pca_list = []
            pub_errs = []

            sqrt_num_param_list = np.sqrt(np.array(num_param_list))
            num_bases_list = self.num_bases * (sqrt_num_param_list / np.sum(sqrt_num_param_list))
            num_bases_list = num_bases_list.astype(int)

            offset = 0
            for i, num_param in enumerate(num_param_list):
                pub_grad = anchor_grads[:, offset:offset + num_param]
                device = pub_grad.device
                offset += num_param
                num_bases = num_bases_list[i]
                print("PUBLIC GET BASES")
                num_bases, pub_error, pca = get_bases(pub_grad, num_bases)
                pub_errs.append(pub_error)
                print('group wise approx PUBLIC  error pca: %.2f%%' % (100 * pub_error))
                num_bases_list[i] = num_bases
                selected_pca_list.append(torch.from_numpy(pca.components_).to(device))
                del pub_grad, pub_error, pca

            self._selected_pca_list = selected_pca_list
            self.num_bases_list = num_bases_list
            self._approx_errors_pca_public = pub_errs
            del pub_errs, num_bases_list

    # @profile
    def forward(self, target_grad, logging=True):

        grad = target_grad

        get_bases(grad, self.num_bases)

        selected_pca: torch.tensor = self._selected_pca_list[i].squeeze().T

        embedding_by_pca = torch.matmul(grad, selected_pca)

        pca_reconstruction_error = check_approx_error(selected_pca, grad)

        if logging:
            cur_approx_pca = torch.matmul(torch.mean(embedding_by_pca, dim=0).view(1, -1),
                                          selected_pca.T).view(-1)
            cur_target = torch.mean(grad, dim=0)
            cur_target_sqr_norm = torch.sum(torch.square(cur_target))

            cur_error_pca = torch.sum(torch.square(cur_approx_pca - cur_target)) / cur_target_sqr_norm
            print('group wise approx PRIVATE error pca: %.2f%%' % (100 * cur_error_pca.item()))

            if i in self.approx_error_pca_private:
                self.approx_error_pca_private[i].append(cur_error_pca.item())
            else:
                self.approx_error_pca_private[i] = []
                self.approx_error_pca_private[i].append(cur_error_pca.item())

        clipped_embedding_pca = clip_column(embedding_by_pca, clip=self.clip0, inplace=False)

        avg_clipped_embedding_pca = torch.sum(clipped_embedding_pca, dim=0) / self.batch_size
        del clipped_embedding_pca

        avg_target_grad = torch.sum(target_grad, dim=0) / self.batch_size

        return avg_clipped_embedding_pca.view(-1), avg_target_grad.view(-1)

    @property
    def selected_pca_list(self):
        return self._selected_pca_list
