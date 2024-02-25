import torch
from private_federated.aggregation_strategies.average_strategy import AverageStrategy
from private_federated.aggregation_strategies.utils import clip_grad_batch


class AverageClipStrategy(AverageStrategy):

    def __init__(self, clip_value: float):
        self._C = clip_value

    def __repr__(self):
        return f"AverageClipStrategy(clip_value={self._C})"

    def __call__(self, grad_batch: torch.tensor) -> torch.tensor:
        clipped_grad_batch = clip_grad_batch(grad_batch, self._C)
        return super().__call__(grad_batch=clipped_grad_batch)
