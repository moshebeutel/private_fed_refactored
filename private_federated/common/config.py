import inspect
from typing import TypeVar

T = TypeVar('T')


def to_dict(T) -> dict:
    members = inspect.getmembers(T, lambda a: not (inspect.isroutine(a)))
    members = [(k, v) for (k, v) in members if not k.startswith('_')]
    return dict(members)


class Config:
    LOG2WANDB = True
    EMBED_GRADS = False
    CLIP_VALUE = 0.1
    NOISE_MULTIPLIER = 4.72193  # 'values': [12.79182, 4.72193, 2.01643]

    USE_GP = True
    GP_KERNEL_FUNCTION = 'RBFKernel'
    assert GP_KERNEL_FUNCTION in ['RBFKernel', 'LinearKernel', 'MaternKernel'], \
        f'GP_KERNEL_FUNCTION={GP_KERNEL_FUNCTION} and should be one of RBFKernel, LinearKernel, MaternKernel'
    GP_NUM_GIBBS_STEPS_TRAIN = 5
    GP_NUM_GIBBS_DRAWS_TRAIN = 20
    GP_NUM_GIBBS_STEPS_TEST = 5
    GP_NUM_GIBBS_DRAWS_TEST = 30
    GP_OUTPUTSCALE_INCREASE = 'constant'
    assert GP_OUTPUTSCALE_INCREASE in ['constant', 'increase', 'decrease'], \
        f'GP_OUTPUTSCALE_INCREASE={GP_OUTPUTSCALE_INCREASE} and should be one of constant, increase, decrease'
    GP_OUTPUTSCALE = 8.0
    GP_LENGTHSCALE = 1.0
    GP_PREDICT_RATIO = 0.5
    GP_OBJECTIVE = 'predictive_likelihood'
    assert GP_OBJECTIVE in ['predictive_likelihood', 'marginal_likelihood'], \
        f'GP_OBJECTIVE={GP_OBJECTIVE} and should be one of predictive_likelihood, marginal_likelihood '
