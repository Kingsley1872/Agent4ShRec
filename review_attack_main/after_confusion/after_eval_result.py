import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attack_method = "mix"
dataset = "DatasetName"

if dataset == "DatasetA":
    target_id_list = []
elif dataset == "DatasetB":
    target_id_list = []
else:
    target_id_list = []

avg_rank_diffs = []
hitrate_attacks = []
hitrate_pures = []
for target_id in target_id_list:

    file1 = f"path/to/results/{dataset}/user_item_predictions.csv"
    if attack_method == "method1":
        file2 = f"path/to/results/{dataset}/user_item_predictions_{target_id}.csv"
    else:
        file2 = f"path/to/results/{dataset}/user_item_predictions_{target_id}_{attack_method}.csv"

    review_pure = pd.read_csv(f"path/to/data/{dataset}/review.csv", header=None)
    review_pure.columns = ['user_id', 'item_id', 'review', 'rating']

    data_pure = pd.read_csv(file1)
    data_pure.set_index(data_pure.columns[0], inplace=True)
    data_pure = (data_pure - data_pure.mean(axis=1).values.reshape(-1, 1)) / data_pure.std(axis=1).values.reshape(-1, 1)
    for i in range(len(review_pure)):
        user_id = review_pure.iloc[i, 0]
        item_id = review_pure.iloc[i, 1]
        data_pure.loc[user_id, f'item_{item_id}'] = 0

    data_attack = pd.read_csv(file2)
    data_attack.set_index(data_attack.columns[0], inplace=True)
    data_attack = data_attack.loc[data_pure.index]
    data_attack = (data_attack - data_attack.mean(axis=1).values.reshape(-1, 1)) / data_attack.std(axis=1).values.reshape(-1, 1)
    for i in range(len(review_pure)):
        user_id = review_pure.iloc[i, 0]
        item_id = review_pure.iloc[i, 1]
        data_attack.loc[user_id, f'item_{item_id}'] = 0

    data_pure_rank = data_pure.rank(axis=1, ascending=False)
    data_attack_rank = data_attack.rank(axis=1, ascending=False)

    data_pure_target = data_pure[f'item_{target_id}']
    data_attack_target = data_attack[f'item_{target_id}']

    data_pure_target_rank = data_pure_rank[f'item_{target_id}']
    data_attack_target_rank = data_attack_rank[f'item_{target_id}']

    avg_rank_pure = data_pure_target.mean()
    avg_rank_attack = data_attack_target.mean()

    hitrate_attack = (data_attack_target_rank <= 10).sum() / len(data_attack_target_rank)
    hitrate_pure = (data_pure_target_rank <= 10).sum() / len(data_pure_target_rank)

    avg_rank_diff = avg_rank_attack - avg_rank_pure
    avg_rank_diffs.append(avg_rank_diff)
    hitrate_attacks.append(hitrate_attack)
    hitrate_pures.append(hitrate_pure)

avg_rank_diff_mean = sum(avg_rank_diffs) / len(avg_rank_diffs)
hitrate_attack_mean = sum(hitrate_attacks) / len(hitrate_attacks)
hitrate_pure_mean = sum(hitrate_pures) / len(hitrate_pures)
print("attack_method:", attack_method)
print(f"Average Rank Difference: {avg_rank_diff_mean}")
print(f"Average Hitrate Attack: {hitrate_attack_mean}")
print(f"Average Hitrate Pure: {hitrate_pure_mean}")
