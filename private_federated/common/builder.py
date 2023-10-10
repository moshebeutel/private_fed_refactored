from private_federated.aggregation_strategies.average_clip_strategy import AverageClipStrategy
from private_federated.aggregation_strategies.average_strategy import AverageStrategy
from private_federated.data.dataset_factory import DatasetFactory
from private_federated.data.loaders_generator import DataLoadersGenerator
from private_federated.data.utils import create_loader_from_dataset
from private_federated.differential_privacy.dp_sgd.dp_sgd_aggregation_starategy import DpSgdAggregationStrategy
from private_federated.differential_privacy.utils import get_sigma
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
    # avg_agg_strategy = AverageStrategy()
    # avg_agg_strategy = AverageClipStrategy(clip_value=0.01)

    import csv
    s = 'q,T,desired_eps,delta,rgp,sigma,calculated_epsilon'
    with open('sigmas.csv', 'w', newline='') as csvfile:
        fieldnames = ['q', 'T', 'desired_eps', 'delta', 'rgp', 'sigma', 'calculated_epsilon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rgp in [False, True]:
            for desired_eps in [1, 3, 8]:
                for num_rounds in [1, 100, 300, 1000]:
                    for sampling_prob in [0.02, 0.1, 0.2, 150.0 / 500.0, 1.0]:
                        for delta in [1e-5, 1e-4, 1e-3]:
                            sigma, calculated_epsilon = get_sigma(q=sampling_prob, T=num_rounds, eps=desired_eps,
                                                                  delta=delta, rgp=rgp)
                            s += f'\n{sampling_prob},{num_rounds},{desired_eps},{delta},{rgp},{sigma},{calculated_epsilon}'
                            writer.writerow({'q': sampling_prob, 'T': num_rounds, 'desired_eps':desired_eps, 'delta': delta, 'rgp': rgp, 'sigma': sigma, 'calculated_epsilon': calculated_epsilon})

    print(s)

    raise Exception

    avg_agg_strategy = DpSgdAggregationStrategy(clip_value=0.0001, sigma=3.776479532659047)
    server = Server(clients=clients_factory.clients, net=net,
                    val_loader=server_val_loader, test_loader=server_test_loader,
                    aggregating_strategy=avg_agg_strategy)
    return server
