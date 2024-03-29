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
