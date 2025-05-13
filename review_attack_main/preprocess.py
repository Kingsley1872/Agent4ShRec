import argparse
import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import json

os.chdir(sys.path[0])


def process_dataset(json_path, select_cols, csv_path):
    print('#### Reading the JSON file...')
    dataset = csv_path.split('/')[-1]
    mapping_file = f"{dataset}_id_update.npy"
    if os.path.exists(mapping_file):
        uid_mapping_dict = np.load(mapping_file, allow_pickle=True)[0]
        iid_mapping_dict = np.load(mapping_file, allow_pickle=True)[1]

    json_path = os.path.join(dataset, json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    df = df[select_cols]

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[(df['timestamp'] >= pd.Timestamp()) & (df['timestamp'] <= pd.Timestamp()))]
    df = df.drop('timestamp', axis=1)
    df.columns = ['userID', 'itemID', 'review', 'rating']
    
    df = df[df['userID'].map(uid_mapping_dict.get).notnull()]
    df = df[df['itemID'].map(iid_mapping_dict.get).notnull()]
    
    df['userID'] = df['userID'].map(uid_mapping_dict.get)
    df['itemID'] = df['itemID'].map(iid_mapping_dict.get)

    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()

    with open('stopwords.txt') as f:
        stop_words = set(f.read().splitlines())
    with open('punctuations.txt') as f:
        punctuations = set(f.read().splitlines())

    def clean_review(review):
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ')
        review = WordPunctTokenizer().tokenize(review)
        review = [word for word in review if word not in stop_words]
        return ' '.join(review)

    df = df.drop(df[[not isinstance(x, str) or len(x) == 0 for x in df['review']]].index)
    df['review'] = df['review'].apply(clean_review)

    os.makedirs(csv_path, exist_ok=True)
    df.to_csv(os.path.join(dataset, 'review.csv'), index=False, header=False, escapechar='\\')
    print(f'#### Total: {len(df)} reviews, {len(df.groupby("userID"))} users, {len(df.groupby("itemID"))} items.')

if __name__ == '__main__':
    dataset = 'Dataset_Name'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path',
                        default=f'{dataset}.jsonl',
                        help='Selected columns of the dataset in JSON format.')
    parser.add_argument('--select_cols', dest='select_cols', nargs='+',
                        default=['user_id', 'parent_asin', 'text', 'rating', 'timestamp'])
    parser.add_argument('--save_dir', dest='save_dir', default=f'data/{dataset}')
    args = parser.parse_args()

    start_time = time.perf_counter()
    process_dataset(args.data_path, args.select_cols, args.save_dir)
    end_time = time.perf_counter()
    print(f'## preprocess.py: Data loading complete! Time used {end_time - start_time:.0f} seconds.')
