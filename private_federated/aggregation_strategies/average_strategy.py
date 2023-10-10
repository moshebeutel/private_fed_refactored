import torch


class AverageStrategy:
    def __call__(self, grad_batch: torch.tensor) -> torch.tensor:
        assert grad_batch.dim() > 1, f'Expected a batch of grads. Got a tensor of shape {grad_batch.shape}'
        return torch.mean(grad_batch, dim=0)
