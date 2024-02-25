import logging
import torch
from private_federated.aggregation_strategies.average_clip_strategy import AverageClipStrategy
from private_federated.aggregation_strategies.utils import clip_grad_batch


class DpSgdAggregationStrategy(AverageClipStrategy):

    def __init__(self, clip_value: float, noise_multiplier: float):
        super().__init__(clip_value=clip_value)
        self._noise_multiplier = noise_multiplier
        self._noise_std = noise_multiplier * clip_value

    def __repr__(self):
        return f"DpSgdAggregationStrategy(clip_value={self._C}, noise_multiplier={self._noise_multiplier})"

    def __call__(self, grad_batch: torch.tensor) -> torch.tensor:
        dp_noise = torch.normal(mean=0, std=self._noise_std,
                                size=grad_batch.size(), device=grad_batch.device)

        clipped_grad_batch = clip_grad_batch(grad_batch, self._C)

        logging.debug(f'dp_noise shape: {dp_noise.shape}'
                      f' clipped grads shape: {clipped_grad_batch.shape}'
                      f' grad_batch shape: {grad_batch.shape}')

        logging.debug(f'dp_noise mean norm: {torch.mean(torch.norm(dp_noise, dim=1))}'
                      f' clipped grads mean norm: {torch.mean(torch.norm(clipped_grad_batch, dim=1))}'
                      f' grad_batch mean norm: {torch.mean(torch.norm(grad_batch, dim=1))}')

        logging.debug(f'mean dp_noise max : {torch.absolute(torch.mean(dp_noise, dim=0)).max()}'
                      f' mean clipped grads max: {torch.absolute(torch.mean(clipped_grad_batch, dim=0)).max()}'
                      f' mean grad_batch max: {torch.absolute(torch.mean(grad_batch, dim=0)).max()}')

        average_clipped_grads = torch.mean(clipped_grad_batch + dp_noise, dim=0)

        return average_clipped_grads
