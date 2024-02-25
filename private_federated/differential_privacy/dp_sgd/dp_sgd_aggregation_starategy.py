import logging

import torch
from private_federated.aggregation_strategies.average_clip_strategy import AverageClipStrategy


class DpSgdAggregationStrategy(AverageClipStrategy):

    def __init__(self, clip_value: float, noise_multiplier: float):
        super().__init__(clip_value=clip_value)
        self._noise_multiplier = noise_multiplier
        self._noise_std = noise_multiplier * clip_value

    def __repr__(self):
        return f"DpSgdAggregationStrategy(clip_value={self._C}, noise_multiplier={self._noise_multiplier})"

    def __call__(self, grad_batch: torch.tensor) -> torch.tensor:
        average_clipped_grads = super().__call__(grad_batch=grad_batch)
        dp_noise = torch.normal(mean=0, std=self._noise_std,
                                size=average_clipped_grads.size(), device=grad_batch.device) / grad_batch.shape[0]
        logging.debug(f'dp_noise shape: {dp_noise.shape}'
                      f' average clipped grads shape: {average_clipped_grads.shape}'
                      f' grad_batch shape: {grad_batch.shape}')

        logging.debug(f'dp_noise norn: {torch.norm(dp_noise)}'
                      f' average clipped grads norm: {torch.norm(average_clipped_grads)}'
                      f' grad_batch norm: {torch.norm(grad_batch)}')

        return average_clipped_grads + dp_noise
