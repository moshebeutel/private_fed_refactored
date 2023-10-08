from torch.utils.data import Dataset

from private_federated.data.utils import gen_random_loaders


class DataLoadersGenerator:
    CLASSES_PER_USER = 10
    BATCH_SIZE = 16

    def __init__(self, users: list[int], datasets: list[Dataset]):
        loaders, cls_partitions = gen_random_loaders(num_users=len(users),
                                                     bz=DataLoadersGenerator.BATCH_SIZE,
                                                     classes_per_user=DataLoadersGenerator.CLASSES_PER_USER,
                                                     datasets=datasets)

        # self._users_loaders = {user: {'train': train_loader, 'validation': validation_loader, 'test': test_loader}
        #                        for user, train_loader, validation_loader, test_loader in
        #                        zip(users, loaders[0], loaders[1], loaders[2])}
        self._users_loaders = {user: {'train': train_loader}
                               for user, train_loader in
                               zip(users, loaders[0])}
        self._users_class_partitions = {user: (cls, prb) for (user, cls, prb) in
                                        zip(users, cls_partitions['class'], cls_partitions['prob'])}

    @property
    def users_loaders(self):
        return self._users_loaders

    @property
    def users_class_partitions(self):
        return self._users_class_partitions
