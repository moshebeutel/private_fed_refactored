import logging
from torch.utils.data import Dataset, DataLoader


class DataLoadersGenerator:
    BATCH_SIZE = 64
    PIN_MEMORY = False
    NUM_WORKERS = 2

    def __init__(self, users_datasets: dict[str, dict[str, Dataset]]):
        loader_params = {"batch_size": DataLoadersGenerator.BATCH_SIZE,
                         "pin_memory": DataLoadersGenerator.PIN_MEMORY,
                         "num_workers": DataLoadersGenerator.NUM_WORKERS}

        self._users_loaders = {user: {split: DataLoader(users_datasets[user][split],
                                                        **{**loader_params, 'shuffle': (split == 'train')})
                                      for split in users_datasets[user]}
                               for user in users_datasets}

        logging.info(f'Generated {list(self._users_loaders.keys())} loaders.')

    @property
    def users_loaders(self):
        return {u: self._users_loaders[u]['train'] for u in self._users_loaders}

    @property
    def users_validation_loaders(self):
        return {u: self._users_loaders[u]['validation'] for u in self._users_loaders}

    @property
    def users_test_loaders(self):
        return {u: self._users_loaders[u]['test'] for u in self._users_loaders}

