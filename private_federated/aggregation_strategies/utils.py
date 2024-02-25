import torch


def flatten_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Flatten a batch of tensors \ grads.
    All dimensions are flattened except the batch dimension
    """
    return batch.reshape(batch.shape[0], -1)


def calc_params_grad_batch_norms(params) -> torch.Tensor:
    assert hasattr(params[0], 'grad_batch'), ('parameters are exoected to have an attribute called grad_batch '
                                              'that stores per gradient samples')

    batch_size = params[0].grad_batch.shape[0]
    grad_norms = torch.zeros(batch_size).to(params[0].device)
    for param in params:
        flat_param = flatten_batch(param.grad_batch)
        param_grad_norm = torch.norm(flat_param, dim=1)
        grad_norms += torch.square(param_grad_norm)
    params_grad_batch_norms = torch.sqrt(grad_norms)
    return params_grad_batch_norms


def clip_params_grad_batch(params, clip_value: float):
    assert hasattr(params[0], 'grad_batch'), ('parameters are exoected to have an attribute called grad_batch '
                                              'that stores per gradient samples')

    batch_size = params[0].grad_batch.shape[0]
    grad_norms: torch.Tensor = calc_params_grad_batch_norms(params)
    scaling_factors: torch.Tensor = clip_value / grad_norms
    scaling_factors[scaling_factors > 1.0] = 1.0

    for param in params:
        p_dim = len(param.shape)
        scaling = scaling_factors.view([batch_size] + [1]*p_dim)
        param.grad_batch *= scaling
        param.grad = torch.mean(param.grad_batch, dim=0)
        param.grad_batch.mul_(0.)


def clip_grad_batch(grad_batch: torch.Tensor, clip_value: float) -> torch.Tensor:
    batch_norms: torch.Tensor = torch.norm(grad_batch, dim=1, keepdim=True)
    scaling_factors: torch.Tensor = batch_norms / clip_value
    scaling_factors = torch.maximum(scaling_factors, torch.ones_like(scaling_factors))
    return torch.div(grad_batch, scaling_factors)

