from pathlib import Path
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms


class DatasetFactory:
    DATASETS_HUB = {'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100}
    DATASETS_DIR = f"{str(Path.home())}/datasets/"
    NORMALIZATIONS = {'CIFAR10': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      'CIFAR100': transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}

    def __init__(self, dataset_name):
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

        self._test_set = dataset_ctor(
            root=dataset_dir,
            train=False,
            download=True,
            transform=transform
        )

        val_size = len(self._test_set)  # 10000
        train_size = len(dataset) - val_size
        self._train_set, self._val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set

    @property
    def test_set(self):
        return self._test_set
