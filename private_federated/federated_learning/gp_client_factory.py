from private_federated.federated_learning.clients_factory import ClientFactory
from private_federated.data.loaders_generator import DataLoadersGenerator


class GPClientFactory(ClientFactory):
    def _create_clients(self, loaders_generator: DataLoadersGenerator):
        from private_federated.federated_learning.gp_client import GPClient
        loaders = loaders_generator.users_loaders
        eval_loaders = loaders_generator.users_test_loaders
        self._clients = [GPClient(cid=cid, loader=loaders[cid], eval_loader=eval_loaders[cid]) for cid in loaders]
