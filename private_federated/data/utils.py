import torch
from torch.utils.data import Dataset
from private_federated.data.random_data_split import gen_classes_per_node, gen_data_split


def gen_random_subsets(num_users: int, classes_per_user: int, datasets: list[Dataset]):
    """
    A variant of the `gen_random_loaders` that originally generates train/val/test loaders of each client
    Taken from https://github.com/AvivSham/pFedHN.git
    I needed a function that returns subset for each client.

    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param datasets: The full datasets to be partitioned to clients
    :return: tuple of:
             train/val/test subset of each client - list of pytorch dataset,
             list of numpy arrays
    """
    subsets_list = []
    for i, d in enumerate(datasets):
        # ensure same partition for train/test/val
        if i == 0:
            # train set
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
        # create subsets for each client
        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
        subsets_list.append(subsets)
    return subsets_list, cls_partitions
