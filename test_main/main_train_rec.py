# -*- coding: utf-8 -*-
import sys
import argparse
import os

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from utils.load_data.load_data import load_data
from model.trainer_rec import rec_trainer
from model.trainer_rec_surprise import basic_rec
from utils.attack.data_to_file import target_prediction_writer

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_rec(data_set_name, model_name, attack_method, target_id, is_train, is_text_init):
    """
    Train a recommendation model with or without attack methods.

    Args:
        data_set_name (str): Name of the dataset.
        model_name (str): Name of the recommendation model.
        attack_method (str): Attack method to be applied.
        target_id (int): Target user/item ID for the attack.
        is_train (int): Flag indicating whether to train the model (1) or not (0).
    """
    is_text_pure = False
    if attack_method == "no":
        attack_method = ""
        model_path = f"result/model_ckpt/{model_name}_{data_set_name}.ckpt"
    else:
        model_path = f"result/model_ckpt/{model_name}_{data_set_name}_{attack_method}.ckpt"

    path_train = f"data/data_attacked/{data_set_name}_{target_id}_{attack_method}.dat"
    path_test = f"data/{data_set_name}/{data_set_name}_test.dat"

    if attack_method == "":
        path_train = f"data/{data_set_name}/{data_set_name}_train.dat"
    if attack_method.split("_")[0] != "llm":
        is_text_pure = True

    dataset_class = load_data(
        path_train=path_train,
        path_test=path_test,
        header=['user_id', 'item_id', 'rating'],
        sep='\t',
        print_log=True
    )

    if model_name in ["NNMF"]:
        predictions, hit_ratios = rec_trainer(model_name, dataset_class, target_id, is_train, model_path, is_text_init, is_text_pure)
    else:
        predictions, hit_ratios = basic_rec(model_name, path_train, path_test, target_id)

    dst_path = f"result/pred_result/{model_name}_{data_set_name}_{target_id}_{str(is_text_init)}_{attack_method}".strip('_')
    target_prediction_writer(predictions, hit_ratios, dst_path)


def parse_arg():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Dataset', help='Input dataset name')
    parser.add_argument('--model_name', type=str, default='Model', help='Model name')
    parser.add_argument('--is_text_init', type=bool, default=False)
    parser.add_argument('--attack_method', type=str, default='no,llm', help='Attack method')

    if parser.get_default('dataset') == "Dataset1":
        parser.add_argument('--target_ids', type=str, default="", help='Comma-separated list of target IDs')
        parser.add_argument('--attack_num', type=int, default=100, help='Number of fake users for the attack')
        parser.add_argument('--filler_num', type=int, default=10, help='Number of filler items per fake user')
    elif parser.get_default('dataset') == "Dataset2":
        parser.add_argument('--target_ids', type=str, default='', help='Comma-separated list of target IDs')
        parser.add_argument('--attack_num', type=int, default=150, help='Number of fake users for the attack')
        parser.add_argument('--filler_num', type=int, default=12, help='Number of filler items per fake user')
    else:
        parser.add_argument('--target_ids', type=str, default='', help='Comma-separated list of target IDs')
        parser.add_argument('--attack_num', type=int, default=80, help='Number of fake users for the attack')
        parser.add_argument('--filler_num', type=int, default=7, help='Number of filler items per fake user')

    args = parser.parse_args()
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    args = parse_arg()

    attack_methods = args.attack_method.split(',')
    for attack_method in attack_methods:
        if attack_method == 'no':
            attack_method_ = attack_method
        else:
            attack_method_ = '_'.join([attack_method, str(args.attack_num), str(args.filler_num)])

        is_train = 1
        if args.model_name != "NNMF":
            args.is_text_init = False
        train_rec(args.dataset, args.model_name, attack_method_, args.target_ids[0], is_train=is_train, is_text_init=args.is_text_init)

        for target in args.target_ids[1:]:
            if attack_method == 'no':
                is_train = 0
            train_rec(args.dataset, args.model_name, attack_method_, target, is_train=is_train, is_text_init=args.is_text_init)
