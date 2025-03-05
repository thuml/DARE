import os

import sys

import numpy as np

from dataset import SeqRecDataset
from train_pytorch import create_model
from tools.logger import CompleteLogger
import logging
import torch


def collect_repr(
        test_dataset,
        args,
):
    model_type = args.long_model_type
    best_model_path = os.path.join(args.model_log_dir, 'model', 'best_{}.pth'.format(model_type))

    model = create_model(model_type, args)

    sys.stdout.flush()
    logging.info('Analysis starts.')
    sys.stdout.flush()

    model.load(best_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset.reset()

    all_target = []
    all_repr = []

    for i, batch_data in enumerate(test_dataset):
        all_target.append(batch_data['target'][:, 0])
        all_repr.append(model.forward(batch_data, mode="evaluate", only_return_representation=True).detach().cpu().numpy())

        if i % 100 == 0:
            logging.info('Processing batch {}'.format(i))

    all_target = np.concatenate(all_target, axis=0)
    all_repr = np.concatenate(all_repr, axis=0)

    os.makedirs(os.path.join(args.log_dir, 'repr'), exist_ok=True)
    np.save(os.path.join(args.log_dir, 'repr', 'target.npy'), all_target)
    np.save(os.path.join(args.log_dir, 'repr', 'repr.npy'), all_repr)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode")

    # dataset config
    parser.add_argument('-train_dataset_path', type=str, default=None)
    parser.add_argument('-val_dataset_path', type=str, default=None)
    parser.add_argument('-test_dataset_path', type=str, default=None)
    parser.add_argument("-max_length", type=int, default=200)
    parser.add_argument('-item_n', type=int, default=4068791)
    parser.add_argument('-cate_n', type=int, default=9408)

    # model config
    parser.add_argument("-short_seq_split", type=str, default=None)
    parser.add_argument("-long_seq_split", type=str, default=None)
    parser.add_argument("-short_model_type", type=str)
    parser.add_argument("-long_model_type", type=str)
    parser.add_argument('-attn_func', type=str, default='learnable')
    parser.add_argument("-hard_or_soft", type=str, default='soft')
    parser.add_argument("-top_k", type=int, default=20)
    parser.add_argument("-use_time", action="store_false")
    parser.add_argument("-use_time_mode", type=str, default='concat')

    parser.add_argument("-model_name", type=str, default="DARE")

    parser.add_argument("-use_aux_loss", type=bool, default=False)
    parser.add_argument('-use_long_seq_average', action="store_true")
    parser.add_argument('-mlp_position_after_concat', action="store_false")
    parser.add_argument("-avoid_domination", action='store_true')
    parser.add_argument("-only_odot", action='store_true')
    parser.add_argument('-use_cross_feature', type=bool, default=False)

    # hyperparameter
    parser.add_argument("-epoch", type=int, default=1)
    parser.add_argument("-batch_size", type=int, default=256)
    parser.add_argument("-category_embedding_dim", type=int, default=16)
    parser.add_argument("-item_embedding_dim", type=int, default=-1)
    parser.add_argument("-time_embedding_dim", type=int, default=4)
    parser.add_argument("-attention_time_embedding_dim", type=int, default=-1)
    parser.add_argument("-attention_category_embedding_dim", type=int, default=-1)
    parser.add_argument("-attention_item_embedding_dim", type=int, default=-1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-weight_decay', type=float, default=1e-6)
    parser.add_argument("-seed", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("-mlp_hidden_layer", type=int, nargs='+', default=[32])

    # logging
    parser.add_argument("-level", type=str, default='INFO')
    parser.add_argument('-log_dir', type=str, default='log')
    parser.add_argument("-test_interval", type=int, default=1000)
    parser.add_argument("-log_interval", type=int, default=100)
    parser.add_argument("-stor_grad", action='store_true')
    parser.add_argument("-observe_attn_repr_grad", action='store_true')
    parser.add_argument('-model_log_dir', type=str, default='analysis_result')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger = CompleteLogger(args.log_dir)
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        level=args.level, stream=sys.stderr)
    logging.info(args)

    test_dataset_path = args.test_dataset_path
    batch_size = args.batch_size
    test_dataset = SeqRecDataset(
        test_dataset_path,
        batch_size,
        max_length=args.max_length,
        apply_hard_search= (args.hard_or_soft == "hard")
    )
    collect_repr(test_dataset, args)

    logger.close()


if __name__ == '__main__':
    main()
