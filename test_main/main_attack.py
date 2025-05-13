import sys
import argparse
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.load_data.load_data import load_data
from utils.load_data.load_attack_info import load_attack_info
from model.attack_model.all_attack_method import all_Attack
from utils.attack.data_to_file import attacked_file_writer


def get_data(data_set_name):
    path_train = f'data/{data_set_name}/' + data_set_name + '_train.dat'
    path_test = f'data/{data_set_name}/' + data_set_name + '_test.dat'
    dataset_class = load_data(
        path_train=path_train,
        path_test=path_test,
        header=['user_id', 'item_id', 'rating'],
        sep='\t',
        print_log=False
    )
    attack_info_path = [
        f"data/{data_set_name}/" + data_set_name + "_selected_items",
        f"data/{data_set_name}/" + data_set_name + "_target_users"
    ]
    attack_info = load_attack_info(*attack_info_path)
    return dataset_class, attack_info


def execute_attack(dataset_class, attack_info, attack_method, target_id, bandwagon_selected, exposure_num, dataset_name, fixed_filler_indicator=None):
    selected_ids, target_users = attack_info[target_id]
    attack_model, attack_num, filler_num = attack_method.split('_')
    attack_num, filler_num = int(attack_num), int(filler_num)

    global_mean, global_std, item_means, item_stds = dataset_class.get_all_mean_std()
    attacker = all_Attack(
        attack_num, filler_num, dataset_class.n_items, target_id,
        global_mean, global_std, item_means, item_stds, 5.0, 1.0, exposure_num,
        fixed_filler_indicator=fixed_filler_indicator
    )

    fake_profiles = None
    if attack_model == "random":
        fake_profiles = attacker.RandomAttack()
    elif attack_model == "bandwagon":
        fake_profiles = attacker.BandwagonAttack(bandwagon_selected)
    elif attack_model == "average":
        fake_profiles = attacker.AverageAttack()
    elif attack_model == "segment":
        fake_profiles = attacker.SegmentAttack(selected_ids)
    elif attack_model == "AoP":
        fake_profiles = attacker.AoPAttack(dataset_name)
    elif attack_model == "llm":
        fake_profiles = attacker.LLMAttack(dataset_name)
    elif attack_model == "mix":
        fake_profiles = attacker.MixAttack(bandwagon_selected, selected_ids)
    else:
        print('Invalid attack method')
        exit()
    return fake_profiles


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='dataset_name', help='Dataset name')
    parser.add_argument('--attack_methods', type=str, default='llm', help='Attack methods')
    parser.add_argument('--targets', type=str, default='', help='Target IDs')
    parser.add_argument('--attack_num', type=int, default=100, help='Number of attackers')
    parser.add_argument('--exposure_num', type=int, default=3, help='Exposure number')
    parser.add_argument('--filler_num', type=int, default=10, help='Number of filler items')
    parser.add_argument('--bandwagon_selected', type=str, default='100', help='Bandwagon selected items')

    args = parser.parse_args()
    args.attack_methods = args.attack_methods.split(',')
    args.targets = list(map(int, args.targets.split(',')))
    args.bandwagon_selected = list(map(int, args.bandwagon_selected.split(',')))
    return args


if __name__ == '__main__':
    args = parse_arg()
    dataset_class, attack_info = get_data(args.dataset)

    for target_id in args.targets:
        try:
            for attack_method in args.attack_methods:
                attack_model = '_'.join([attack_method, str(args.attack_num), str(args.filler_num)])
                ori_path = f'data/{args.dataset}/' + args.dataset + '_train.dat'
                dst_path = f"data/data_attacked/" + '_'.join([args.dataset, str(target_id), attack_model]) + ".dat"

                if attack_method == 'llm':
                    fake_profiles, fake_reviews = execute_attack(
                        dataset_class, attack_info, attack_model, target_id,
                        args.bandwagon_selected, args.exposure_num, args.dataset, None
                    )
                    fake_reviews_path = f"data/data_attacked/" + '_'.join([args.dataset, str(target_id), attack_model]) + "_fake_reviews.npy"
                    np.save(fake_reviews_path, fake_reviews)
                else:
                    fake_profiles = execute_attack(
                        dataset_class, attack_info, attack_model, target_id,
                        args.bandwagon_selected, args.exposure_num, args.dataset, None
                    )
                attacked_file_writer(ori_path, dst_path, fake_profiles, dataset_class.n_users)

            print(f"Target ID {target_id} attack completed")
        except Exception as e:
            print(f"Target ID {target_id} attack failed: {e}")
            continue
