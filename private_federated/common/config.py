import inspect


def to_dict() -> dict:
    members = inspect.getmembers(Config, lambda a: not (inspect.isroutine(a)))
    members = [(k, v) for (k, v) in members if not k.startswith('_')]
    return dict(members)


class Config:
    NUM_ROUNDS = 20
    EMBED_GRADS = False
