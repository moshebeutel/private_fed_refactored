import logging
from private_federated.federated_learning.server import Server
from private_federated.data.dataset_factory import DatasetFactory
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.federated_learning.client import Client
from typing import TypeVar, Generic

T = TypeVar('T')


class ClientFactory(Generic[T]):
    NUM_CLIENTS_PUBLIC = 100
    NUM_CLIENTS_PRIVATE = 1000
    NUM_CLIENTS_VAL = 50
    NUM_CLIENTS_TEST = 200
    NUM_ALL_USERS = 1350

    def __init__(self, dataset_factory: DatasetFactory):
        assert ClientFactory.NUM_CLIENTS_PRIVATE >= Server.NUM_CLIENT_AGG, \
            f'Cant aggregate {Server.NUM_CLIENT_AGG} out of {ClientFactory.NUM_CLIENTS_PRIVATE} train users'

        num_active_users = (ClientFactory.NUM_CLIENTS_PUBLIC +
                            ClientFactory.NUM_CLIENTS_PRIVATE +
                            ClientFactory.NUM_CLIENTS_VAL +
                            ClientFactory.NUM_CLIENTS_TEST)
        num_dummy_users = ClientFactory.NUM_ALL_USERS - num_active_users
        assert num_dummy_users >= 0, (f'Expected num active users be no more than {ClientFactory.NUM_ALL_USERS}.'
                                      f' Got {num_active_users} active users and {num_dummy_users} dummy users')

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

        self.all_users_list = list(self.public_users +
                                   self.train_user_list +
                                   self.validation_user_list +
                                   self.test_user_list +
                                   self.dummy_users)
        assert len(set(self.all_users_list)) == len(
            self.all_users_list), f"duplicate users found: {self.all_users_list}"

        loaders_generator = DataLoadersGenerator(users=self.all_users_list, datasets=[dataset_factory.train_set,
                                                                                      dataset_factory.val_set,
                                                                                      dataset_factory.test_set])

        self._create_clients(loaders_generator)
        self._public_clients = [c for c in self._clients if c.cid in self.public_users]
        self._private_train_clients = [c for c in self._clients if c.cid in self.train_user_list]
        self._validation_clients = [c for c in self._clients if c.cid in self.validation_user_list]
        self._test_clients = [c for c in self._clients if c.cid in self.test_user_list]

        ClientFactory.log_user_list('Public Users', self.public_users)
        ClientFactory.log_user_list('Private Users', self.train_user_list)
        ClientFactory.log_user_list('Validation Users', self.validation_user_list)
        ClientFactory.log_user_list('Test Users', self.test_user_list)
        ClientFactory.log_user_list('Dummy Users', self.dummy_users)
        ClientFactory.log_user_list('All Users', self.all_users_list)

    def _get_client_type(self) -> T:
        raise NotImplementedError

    def _create_clients(self, loaders_generator):
        train_loaders = loaders_generator.users_loaders
        eval_loaders = loaders_generator.users_test_loaders
        loaders = {cid: {'train': train_loaders[cid], 'eval': eval_loaders[cid]} for cid in train_loaders}
        self._clients = [self._get_client_type()(cid=cid, train_loader=loaders[cid]['train'], eval_loader=loaders[cid]['eval'])
                         for cid in loaders]

    @staticmethod
    def log_user_list(list_name: str, user_list: list[str]):
        log_str = f"{list_name}: {user_list[0]}-{user_list[-1]}" if len(user_list) > 0 else f"{list_name} is empty"
        logging.info(log_str)

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


class NetClientsFactory(ClientFactory[Client]):
    def _get_client_type(self):
        return Client
