# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys, argparse
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_data.load_data import load_data
from utils.load_data.load_attack_info import load_attack_info

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def attack_evaluate(real_preds_path, attacked_preds_file, non_rated_users, target_users):
    names = ['uid', 'rating', 'HR_1', 'HR_3', 'HR_5', 'HR_10', 'HR_20', 'HR_50']
    real_preds = pd.read_csv(real_preds_path, sep='\t', names=names, engine='python')
    attacked_preds = pd.read_csv(attacked_preds_file, sep='\t', names=names, engine='python')

    shift_target = np.mean(attacked_preds.iloc[target_users, 1].values - real_preds.iloc[target_users, 1].values)
    shift_all = np.mean(attacked_preds.iloc[non_rated_users, 1].values - real_preds.iloc[non_rated_users, 1].values)

    HR_real_target = real_preds.iloc[target_users, range(2, 8)].mean().values
    HR_real_all = real_preds.iloc[non_rated_users, range(2, 8)].mean().values

    HR_attacked_target = attacked_preds.iloc[target_users, range(2, 8)].mean().values
    HR_attacked_all = attacked_preds.iloc[non_rated_users, range(2, 8)].mean().values

    return shift_target, HR_real_target, HR_attacked_target, shift_all, HR_real_all, HR_attacked_all


def eval_attack(dataset_name, model_name, attack_method, target_id, is_text_init):
    dir = "results/predictions/"
    real_preds_path = dir + '_'.join([model_name, dataset_name, str(target_id)]) + "_" + str(is_text_init)
    attacked_preds_file = real_preds_path + "_" + attack_method

    if dataset_name == 'dataset1':
        path_train = "./data/dataset1/train.dat"
        path_test = "./data/dataset1/test.dat"
        attack_info_path = ["./data/dataset1/selected_items", "./data/dataset1/target_users"]
    elif dataset_name == 'dataset2':
        path_train = "./data/dataset2/train.dat"
        path_test = "./data/dataset2/test.dat"
        attack_info_path = ["./data/dataset2/selected_items", "./data/dataset2/target_users"]
    else:
        path_train = f"./data/{dataset_name}/train.dat"
        path_test = f"./data/{dataset_name}/test.dat"
        attack_info_path = [f"./data/{dataset_name}/selected_items", f"./data/{dataset_name}/target_users"]

    attack_info = load_attack_info(*attack_info_path)
    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=False)

    target_users = attack_info[target_id][1]
    non_rated_users = dataset_class.get_item_nonrated_users(target_id)
    
    res = attack_evaluate(real_preds_path, attacked_preds_file, non_rated_users, target_users)
    
    target, all = res[:3], res[3:]
    target_str = '\t'.join([str(target[0]), '\t'.join(map(str, target[1])), '\t'.join(map(str, target[2]))])
    all_str = '\t'.join([str(all[0]), '\t'.join(map(str, all[1])), '\t'.join(map(str, all[2]))])

    info = '\t'.join([model_name, attack_method, str(target_id)])
    return info + '\t' + target_str + '\t' + all_str


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='dataset1', help='Dataset name')
    parser.add_argument('--attack_num', type=int, default=50, help='Number of attacks')
    parser.add_argument('--filler_num', type=int, default=10, help='Number of fillers')
    parser.add_argument('--attack_methods', type=str, default='method1', help='Comma-separated attack methods')
    parser.add_argument('--is_text_init', type=bool, default=False)
    parser.add_argument('--rec_model_names', type=str, default='model1', help='Comma-separated model names')
    parser.add_argument('--target_ids', type=str, default='', help='Comma-separated target IDs')

    args = parser.parse_args()
    args.attack_methods = args.attack_methods.split(',')
    args.rec_model_names = args.rec_model_names.split(',')
    args.target_ids = list(map(int, args.target_ids.split(',')))
    return args


if __name__ == '__main__':
    args = parse_arg()
    result = []

    for attack_method in args.attack_methods:
        for model_name in args.rec_model_names:
            for target_id in args.target_ids:
                attack_method_ = '_'.join([attack_method, str(args.attack_num), str(args.filler_num)])
                try:
                    result_ = eval_attack(args.dataset, model_name, attack_method_, target_id, args.is_text_init)
                    result.append(result_.split('\t'))
                except Exception as e:
                    print(f"Error: {attack_method}, {model_name}, {target_id}, {e}")

    result = np.array(result).transpose()
    result = pd.DataFrame(dict(zip(range(result.shape[0]), result)))
    result.columns = ['model_name', 'attack_method', 'target_id', 'shift_target', 'HR_1_real_target', 'HR_3_real_target', 
                      'HR_5_real_target', 'HR_10_real_target', 'HR_20_real_target', 'HR_50_real_target', 
                      'HR_1_attacked_target', 'HR_3_attacked_target', 'HR_5_attacked_target', 'HR_10_attacked_target', 
                      'HR_20_attacked_target', 'HR_50_attacked_target', 'shift_all', 'HR_1_real_all', 'HR_3_real_all', 
                      'HR_5_real_all', 'HR_10_real_all', 'HR_20_real_all', 'HR_50_real_all', 'HR_1_attacked_all', 
                      'HR_3_attacked_all', 'HR_5_attacked_all', 'HR_10_attacked_all', 'HR_20_attacked_all', 
                      'HR_50_attacked_all']
    
    numeric_columns = result.columns[3:]
    averages = result[numeric_columns].astype(float).mean(axis=0)
    print(averages)
    result.to_csv(args.dataset + '_performance_all.csv', index=False)
