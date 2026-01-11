import torch


def clip_grad_batch(grad_batch: torch.Tensor, clip_value: float) -> torch.Tensor:
    """
    Clip batch gradients to a given value
    :param grad_batch: The batch gradients
    :param clip_value: clip value
    :return: The clipped batch gradients
    """
    batch_norms: torch.Tensor = torch.linalg.vector_norm(grad_batch, dim=1, keepdim=True)
    scaling_factors: torch.Tensor = batch_norms / clip_value
    scaling_factors = torch.maximum(scaling_factors, torch.ones_like(scaling_factors))
    return torch.div(grad_batch, scaling_factors)
