import torch
from private_federated.aggregation_strategies.average_clip_strategy import AverageClipStrategy


class DpSgdAggregationStrategy(AverageClipStrategy):
    # def __init__(self, clip_value: float, sigma: float):
    #     super().__init__(clip_value=clip_value)
    #     self._noise_std = sigma * clip_value

    def __init__(self, dp_claculator):
        super().__init__(clip_value=dp_claculator.average_privacy_param.sensitivity)
        self._noise_std = (dp_claculator.average_privacy_param.gaussian_standard_deviation
                           / dp_claculator.average_privacy_param.sensitivity)
        self._dp_calc = dp_claculator

    def __call__(self, grad_batch: torch.tensor) -> torch.tensor:
        average_clipped_grads = super().__call__(grad_batch=grad_batch)
        dp_noise = torch.normal(mean=0, std=self._noise_std,
                                size=average_clipped_grads.size(), device=grad_batch.device) / grad_batch.shape[0]
        return average_clipped_grads + dp_noise
