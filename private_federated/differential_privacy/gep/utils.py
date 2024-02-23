from sklearn.decomposition import PCA
import torch


def get_num_bases_for_upper_frac(pca: PCA, frac: float):
    assert 0.0 < frac < 1.0, f'Expected a positive fraction of 1. Got {frac}'
    cumsums = pca.explained_variance_ratio_.cumsum()
    num_bases_for_upper_frac = len(cumsums[cumsums < frac])
    del cumsums
    return num_bases_for_upper_frac


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # print(f'Normalize the {i}-th column')
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


def clip_column(tsr: torch.tensor, clip=1.0, inplace=True):
    if inplace:
        inplace_clipping(tsr, torch.tensor(clip).to(tsr.device))
    else:
        norms = torch.norm(tsr, dim=1)
        scale = torch.clamp(clip / norms, max=1.0)
        return tsr * scale.view(-1, 1)


@torch.jit.script
def inplace_clipping(matrix, clip):
    n, m = matrix.shape
    for i in range(n):
        # Normalize the i'th row
        col = matrix[i:i + 1, :]
        col_norm = torch.sqrt(torch.sum(col ** 2))
        if col_norm > clip:
            col /= (col_norm / clip)


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


def get_approx_grad(embedding, bases_list, num_bases_list):
    grad_list = []
    offset = 0
    if len(embedding.shape) > 1:
        bs = embedding.shape[0]
    else:
        bs = 1
    embedding = embedding.view(bs, -1)

    for i, bases in enumerate(bases_list):
        num_bases = num_bases_list[i]

        bases_components = bases.squeeze() if Config.GEP_USE_PCA else bases.T

        grad = torch.matmul(embedding[:, offset:offset + num_bases].view(bs, -1), bases_components)

        if bs > 1:
            grad_list.append(grad.view(bs, -1))
        else:
            grad_list.append(grad.view(-1))
        offset += num_bases
    if bs > 1:
        return torch.cat(grad_list, dim=1)
    else:
        return torch.cat(grad_list)


def flatten_tensor(tensor_list: list[torch.tensor]) -> torch.tensor:
    """
    Taken from https://github.com/dayu11/Gradient-Embedding-Perturbation
    """
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param
