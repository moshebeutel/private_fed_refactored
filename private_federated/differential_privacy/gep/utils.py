from sklearn.decomposition import PCA
import torch


def check_approx_error(L, target) -> float:
    L = L.to(target.device)
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.T)
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))

    return -1.0 if target.item() == 0 else error.item() / target.item()


def get_bases(pub_grad, num_bases):
    num_k = pub_grad.shape[0]
    num_p = pub_grad.shape[1]

    num_bases = min(num_bases, min(num_p, num_k))

    pca = PCA(n_components=num_bases)
    pca.fit(pub_grad.cpu().detach().numpy())

    error_rate = check_approx_error(torch.from_numpy(pca.components_).T, pub_grad)

    return num_bases, error_rate, pca


def flatten_tensor(tensor_list: list[torch.tensor]) -> torch.tensor:
    """
    Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation
    """
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param
