from private_federated.data.dataset_factory import DatasetFactory
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.data.utils import create_loader_from_dataset
from private_federated.federated_learning.clients_factory import ClientFactory
from private_federated.federated_learning.server import Server
from private_federated.models import model_factory


def build_all(args):
    dataset_factory = DatasetFactory(dataset_name=args.dataset_name)
    users_list = [cid for cid in range(args.num_clients)]
    loaders_generator = DataLoadersGenerator(users=users_list, datasets=[dataset_factory.train_set])
    clients_factory = ClientFactory(loaders=loaders_generator.users_loaders)
    net = model_factory.get_model(args)

    loader_params = {"batch_size": DataLoadersGenerator.BATCH_SIZE, "shuffle": False,
                     "pin_memory": True, "num_workers": 0}
    server_val_loader = create_loader_from_dataset(dataset_factory.val_set, **loader_params)
    server_test_loader = create_loader_from_dataset(dataset_factory.test_set, **loader_params)
    server = Server(clients=clients_factory.clients, net=net,
                    val_loader=server_val_loader, test_loader=server_test_loader)
    return server
