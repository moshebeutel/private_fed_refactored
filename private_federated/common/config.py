import inspect
from typing import TypeVar

T = TypeVar('T')


def to_dict(T) -> dict:
    members = inspect.getmembers(T, lambda a: not (inspect.isroutine(a)))
    members = [(k, v) for (k, v) in members if not k.startswith('_')]
    return dict(members)


class Config:
    LOG2WANDB = False
    EMBED_GRADS = True
    CLIP_VALUE = 0.1
    NOISE_MULTIPLIER = 2.01643  # 'values': [12.79182, 4.72193, 2.01643]
