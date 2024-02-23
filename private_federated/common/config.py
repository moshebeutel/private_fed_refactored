import inspect


def to_dict() -> dict:
    members = inspect.getmembers(Config, lambda a: not (inspect.isroutine(a)))
    members = [(k, v) for (k, v) in members if not k.startswith('_')]
    return dict(members)


class Config:
    LOG2WANDB = False
    EMBED_GRADS = True
    CLIP_VALUE = 0.1
    NOISE_MULTIPLIER = 12.79182
