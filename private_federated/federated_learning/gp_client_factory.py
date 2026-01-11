from private_federated.federated_learning.clients_factory import ClientFactory
from private_federated.federated_learning.gp_client import GPClient


# class GPClientFactory(ClientFactory[GPClient]):
class GPClientFactory(ClientFactory):
    def _get_client_type(self):
        return GPClient
