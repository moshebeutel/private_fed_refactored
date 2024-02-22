import torch

from private_federated.data.dataset_factory import DatasetFactory
from private_federated.federated_learning.client import Client
from private_federated.data.loaders_generator import DataLoadersGenerator


class ClientFactory:
    NUM_CLIENTS_PUBLIC, NUM_CLIENTS_PRIVATE, NUM_CLIENT_AGG = 10, 250, 50
    assert NUM_CLIENTS_PRIVATE >= NUM_CLIENT_AGG, \
        f'Cant aggregate {NUM_CLIENT_AGG} out of {NUM_CLIENTS_PRIVATE} train users'

    NUM_CLIENTS_VAL = 50
    NUM_CLIENTS_TEST = 400
    NUM_ALL_USERS = 900

    def __init__(self, dataset_factory: DatasetFactory):

        num_active_users = (ClientFactory.NUM_CLIENTS_PUBLIC +
                            ClientFactory.NUM_CLIENTS_PRIVATE +
                            ClientFactory.NUM_CLIENTS_VAL +
                            ClientFactory.NUM_CLIENTS_TEST)
        num_dummy_users = ClientFactory.NUM_ALL_USERS - num_active_users
        assert num_dummy_users > 0, f'Expected num active users be less than {ClientFactory.NUM_ALL_USERS}'

        self.train_user_list = [('%d' % i).zfill(4) for i in range(ClientFactory.NUM_CLIENTS_PUBLIC + 1,
                                                                   ClientFactory.NUM_CLIENTS_PUBLIC +
                                                                   ClientFactory.NUM_CLIENTS_PRIVATE + 1)]

        self.public_users = [('%d' % i).zfill(4) for i in range(1, ClientFactory.NUM_CLIENTS_PUBLIC + 1)]

        self.validation_user_list = [('%d' % i).zfill(4) for i in range(ClientFactory.NUM_CLIENTS_PUBLIC +
                                                                        ClientFactory.NUM_CLIENTS_PRIVATE + 1,
                                                                        ClientFactory.NUM_CLIENTS_PUBLIC +
                                                                        ClientFactory.NUM_CLIENTS_PRIVATE
                                                                        + ClientFactory.NUM_CLIENTS_VAL + 1)]

        self.test_user_list = [('%d' % i).zfill(4) for i in range(ClientFactory.NUM_CLIENTS_PUBLIC +
                                                                  ClientFactory.NUM_CLIENTS_PRIVATE +
                                                                  ClientFactory.NUM_CLIENTS_VAL + 1,
                                                                  ClientFactory.NUM_CLIENTS_PUBLIC +
                                                                  ClientFactory.NUM_CLIENTS_PRIVATE +
                                                                  ClientFactory.NUM_CLIENTS_VAL +
                                                                  ClientFactory.NUM_CLIENTS_TEST + 1)]

        self.dummy_users = [('%d' % i).zfill(4) for i in range(ClientFactory.NUM_CLIENTS_PUBLIC +
                                                               ClientFactory.NUM_CLIENTS_PRIVATE +
                                                               ClientFactory.NUM_CLIENTS_VAL +
                                                               ClientFactory.NUM_CLIENTS_TEST + 1,
                                                               ClientFactory.NUM_ALL_USERS + 1)]

        self.all_users_list = list(set(self.public_users +
                                   self.train_user_list +
                                   self.validation_user_list +
                                   self.test_user_list +
                                   self.dummy_users))

        loaders_generator = DataLoadersGenerator(users=self.all_users_list, datasets=[dataset_factory.train_set])
        loaders = loaders_generator.users_loaders
        self._clients = [Client(cid=cid, loader=loaders[cid]['train']) for cid in loaders]
        self._public_clients = [c for c in self._clients if c.cid in self.public_users]
        self._private_train_clients = [c for c in self._clients if c.cid in self.train_user_list]
        self._validation_clients = [c for c in self._clients if c.cid in self.validation_user_list]
        self._test_clients = [c for c in self._clients if c.cid in self.test_user_list]

    @property
    def clients(self):
        return self._clients

    @property
    def public_clients(self):
        return self._public_clients

    @property
    def private_train_clients(self):
        return self._private_train_clients

    @property
    def validation_clients(self):
        return self._validation_clients

    @property
    def test_clients(self):
        return self._test_clients
