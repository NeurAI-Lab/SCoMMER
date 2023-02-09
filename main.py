# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_gcil_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True, help='Model name.',
                        choices=get_all_models())
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--experiment_id', type=str, required=True,
                            help='Experiment identifier.')
        if args.dataset == 'gcil-cifar100':
            add_gcil_args(parser)

        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset]['sgd'][args.weight_dist]
            else:
                best = best_args[args.dataset]['sgd']
        else:
            if args.dataset == 'gcil-cifar100':
                best = best_args[args.dataset][args.model][args.weight_dist]
            else:
                best = best_args[args.dataset][args.model]
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        if args.dataset == 'gcil-cifar100':
            add_gcil_args(parser)
        args = parser.parse_args()
        print(args)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer':
        setattr(args, 'batch_size', 1)

    # save arguments
    model_dir = os.path.join(args.output_dir, "saved_models", args.dataset, args.experiment_id)
    args_path = os.path.join(model_dir, "args.txt")
    os.makedirs(model_dir, exist_ok=True)

    z = vars(args).copy()
    with open(args_path, "w") as f:
        f.write("arguments: " + json.dumps(z) + "\n")

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    params = count_parameters(backbone)
    print('!' * 30)
    print(f'Number of parameters = {params}')
    print('!' * 30)

    results_dir = os.path.join(args.output_dir, "results", dataset.SETTING, args.dataset, model.NAME, args.experiment_id, "mean_accs.csv")

    if os.path.exists(results_dir):
        print('*' * 30)
        print('Experiment Already trained')
        print(results_dir)
        print('*' * 30)

    else:
        if isinstance(dataset, ContinualDataset):
            train(model, dataset, args)
        else:
            assert not hasattr(model, 'end_task')
            ctrain(args)

if __name__ == '__main__':
    main()
