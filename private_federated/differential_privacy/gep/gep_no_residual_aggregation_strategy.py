from aggregation_strategies.average_clip_strategy import AverageClipStrategy
import torch
from differential_privacy.gep.gep import GEP


class GepNoResidualAggregationStrategy(AverageClipStrategy):
    def __init__(self, clip_value: float, noise_multiplier: float, gep: GEP):
        super().__init__(clip_value=clip_value)
        self._noise_std = noise_multiplier * clip_value
        self._gep = gep

    def __call__(self, grad_batch: torch.tensor) -> torch.tensor:
        pass
