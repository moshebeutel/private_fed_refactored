from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
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

    def dataset_ctor(self, ctor_fn, root, train=True, download=True, transform=None):
        return ctor_fn(root, train=train, download=download, transform=transform)

    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set

    @property
    def test_set(self):
        return self._test_set


class PutEMGDataset(Dataset):
    def __init__(self, root: str, user: str, split: str = 'train', device: str = 'cpu'):
        self._root_path: Path = Path(root) / f'{int(user):02}'
        self._root = self._root_path.as_posix()
        assert split in ['train', 'val', 'test'], f'Expected split name one of train, val, test'
        self._split: str = split
        self._X_file_path = self._root_path / f'X_{split}.pt'
        self._y_file_path = self._root_path / f'y_{split}.pt'
        assert self._X_file_path.exists() and self._y_file_path.exists(), f'Expected {self._X_file_path} file and {self._y_file_path} file'
        X = torch.load(self._X_file_path.as_posix(), map_location=torch.device('cpu'), mmap=True)
        y = torch.load(self._y_file_path.as_posix(), map_location=torch.device('cpu'), mmap=True)
        assert X.shape[0] == y.shape[0], 'X and y should have the same number of samples'
        self._len = X.shape[0]
        del X, y
        self._device = device

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        assert 0 < index < self._len, f'Index {index} out of [0, {self._len}]'
        X: Tensor = torch.load(self._X_file_path.as_posix(), map_location=torch.device(self._device))
        y: Tensor = torch.load(self._y_file_path.as_posix(), map_location=torch.device(self._device))
        ret_X: Tensor = torch.clone(X[index])
        ret_y: Tensor = torch.clone(y[index])
        X.cpu(), y.cpu()
        del X, y
        return ret_X, ret_y


class PutEMGDatasetFactory(DatasetFactory):

    def dataset_ctor(self, ctor_fn, root, train=True, download=True, transform=None):
        return ctor_fn(root, train=train, download=download, transform=transform)


if __name__ == '__main__':
    root_path = Path.home() / 'datasets/EMG/putEMG/tensors'
    assert root_path.exists(), f'Expected root path to be {root_path}'
    root = root_path.as_posix()
    ds = PutEMGDataset(root=root, user='3', split='train', device='cuda')

    print(ds.__len__())
    print(ds.__getitem__(1))
    print(len(ds))

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    for X, y in loader:
        print(X.shape, y.shape)
