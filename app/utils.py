import logging
import wandb
from private_federated.common.config import Config


def create_run_name(args):
    run_name = (f'{args.dataset_name},{args.model_name},'
                f'Embed Grads: {args.embed_grads},'
                f'Noise Mult. {args.noise_multiplier},'
                f'Use GP: {args.use_gp},'
                f'Num Clients Agg: {args.num_clients_agg},'
                f'Clip Value {args.clip},'
                f'Client Learning Rate {args.client_learning_rate},'
                f'Server Learning Rate {args.server_learning_rate},'
                f'Internal Epochs {args.clients_internal_epochs}')
    if args.embed_grads:
        run_name += (f','
                     f'Num Basis Elements {args.embedding_num_bases},'
                     f'Grads History Size {args.grads_history_size},'
                     f'Num Public Clients {args.num_clients_public}')

    logging.info('\n'.join(run_name.split(',')))
    if Config.LOG2WANDB:
        wandb.run.name = run_name
