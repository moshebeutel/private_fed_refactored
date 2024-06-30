from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PutEMGDataset(Dataset):
    def __init__(self, root: str, user: str, split: str = 'train', device: str = 'cpu'):
        self._root_path: Path = Path(root) / f'{int(user):02}'
        self._root = self._root_path.as_posix()
        assert split in ['train', 'validation', 'test'], f'Expected split name one of train, val, test. Got {split}'
        self._split: str = split
        self._X_file_path = self._root_path / f'X_{split}_windowed.pt'
        self._y_file_path = self._root_path / f'y_{split}_windowed.pt'
        assert self._X_file_path.exists() and self._y_file_path.exists(), (f'Expected {self._X_file_path} file'
                                                                           f' and {self._y_file_path} file')
        X = torch.load(self._X_file_path.as_posix())
        y = torch.load(self._y_file_path.as_posix())
        assert X.shape[0] == y.shape[0], 'X and y should have the same number of samples'
        self._len = X.shape[0]
        del X, y
        self._device = device

    def __len__(self):
        return self._len

    def __getitem__(self, index: int):
        assert 0 <= index < self._len, f'Index {index} out of [0, {self._len}-1]'
        X: Tensor = torch.load(self._X_file_path.as_posix())
        y: Tensor = torch.load(self._y_file_path.as_posix())
        ret_X: Tensor = torch.clone(X[index]).float()
        ret_y: Tensor = torch.clone(y[index]).long()
        ret_y[ret_y > 5] -= 2
        X.cpu(), y.cpu()
        del X, y
        return ret_X, ret_y
