import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms

from private_federated.common.config import Config
from private_federated.data.put_emg_dataset import PutEMGDataset
from private_federated.data.utils import gen_random_subsets, load_npy


class DatasetFactory:
    DATASETS_HUB = {'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100, 'putEMG': PutEMGDataset}
    DATASETS_DIR = f"{str(Path.home())}/datasets/"
    NORMALIZATIONS = {'CIFAR10': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      'CIFAR100': transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}
    CLASSES_PER_USER = 10

    def __init__(self, dataset_name: str, users):
        assert dataset_name in DatasetFactory.DATASETS_HUB, (f'Expected dataset name one of'
                                                             f' {DatasetFactory.DATASETS_HUB.keys()}.'
                                                             f' Got {dataset_name}')

        normalization = DatasetFactory.NORMALIZATIONS[dataset_name]
        transform = transforms.Compose([transforms.ToTensor(), normalization])

        dataset_ctor = DatasetFactory.DATASETS_HUB[dataset_name]
        dataset_dir = DatasetFactory.DATASETS_DIR + dataset_name
        dataset = dataset_ctor(
            root=dataset_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_set = dataset_ctor(
            root=dataset_dir,
            train=False,
            download=True,
            transform=transform
        )

        val_size = len(test_set)  # 10000
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

        self._classes = dataset.classes

        subsets, cls_partitions = gen_random_subsets(num_users=len(users),
                                                     classes_per_user=DatasetFactory.CLASSES_PER_USER,
                                                     datasets=[train_set, val_set, test_set])

        self._users_subsets = {user: {'train': train_subset, 'validation': validation_subset, 'test': test_subset}
                               for user, train_subset, validation_subset, test_subset in
                               zip(users, subsets[0], subsets[1], subsets[2])}

        self._users_class_partitions = {user: (cls, prb) for (user, cls, prb) in
                                        zip(users, cls_partitions['class'], cls_partitions['prob'])}

    def dataset_ctor(self, ctor_fn, root, train=True, download=True, transform=None):
        return ctor_fn(root, train=train, download=download, transform=transform)

    @property
    def users_subsets(self):
        return self._users_subsets

    @property
    def users_class_partitions(self):
        return self._users_class_partitions

    @property
    def train_set(self):
        return {u: self._users_subsets[u]['train'] for u in self._users_subsets}

    @property
    def val_set(self):
        return {u: self._users_subsets[u]['validation'] for u in self._users_subsets}

    @property
    def test_set(self):
        return {u: self._users_subsets[u]['test'] for u in self._users_subsets}

    @property
    def classes(self):
        return self._classes


class PutEMGDatasetFactory(DatasetFactory):
    def __init__(self, dataset_name: str, users):
        dataset_ctor = DatasetFactory.DATASETS_HUB[dataset_name]
        DatasetFactory.CLASSES_PER_USER = 8
        root_path = Path.home() / 'datasets/EMG/putEMG/windowed'
        assert root_path.exists(), f'Expected root path to be {root_path}'
        # root = root_path.as_posix()
        # self._users_subsets = {user: {split: PutEMGDataset(root=root, user=user, split=split, device='cuda')
        #                               for split in ['train', 'validation', 'test']}
        #                        for user in users}
        logging.info(f'PutEMGDatasetFactory create user subsets')
        self._users_subsets = {
            user:
                {split:
                    TensorDataset(
                        torch.from_numpy(load_npy(root_path / f'{int(user):02}' / f'X_{split}_windowed.npy')).float(),
                        torch.from_numpy(load_npy(root_path / f'{int(user):02}' / f'y_{split}_windowed.npy')).long()
                    )
                    for split in ['train', 'validation', 'test']}
            for user in users}
        logging.info(f'PutEMGDatasetFactory created user partitions for {len(users)} users')

    def dataset_ctor(self, ctor_fn, root, train=True, download=True, transform=None):
        return ctor_fn(root, train=train, download=download, transform=transform)

    @property
    def classes(self):
        return [0, 1, 2, 3, 6, 7, 8, 9]


if __name__ == '__main__':
    root_path = Path.home() / 'datasets/EMG/putEMG/tensors'
    assert root_path.exists(), f'Expected root path to be {root_path}'
    root = root_path.as_posix()
    ds = PutEMGDataset(root=root, user='3', split='train', device=Config.DEVICE)

    print(ds.__len__())
    print(ds.__getitem__(1))
    print(len(ds))

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    for X, y in loader:
        print(X.shape, y.shape)
