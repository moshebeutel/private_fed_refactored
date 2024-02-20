import torch
from private_federated.aggregation_strategies.average_strategy import AverageStrategy


class AverageClipStrategy(AverageStrategy):
    def __init__(self, clip_value: float):
        self._C = clip_value

    def __call__(self, grad_batch: torch.tensor) -> torch.tensor:
        flat_g = grad_batch.reshape(grad_batch.shape[0], -1)
        norm = torch.norm(flat_g, dim=1)
        norm_ratio = norm / self._C
        clip_factor = torch.clip(norm_ratio, min=1.0)
        clipped_grad_batch = torch.div(flat_g, clip_factor.reshape(-1, 1)).reshape(grad_batch.shape)
        return super().__call__(grad_batch=clipped_grad_batch)

