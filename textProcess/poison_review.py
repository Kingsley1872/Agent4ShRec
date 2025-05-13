import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
import math
import pickle
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_bert_embedding(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].mean(dim=0).numpy()

if __name__ == "__main__":
    num_user = 2700
    attack = True

    text_train_df = pd.read_csv("data/train.csv", header=None)
    user_reviews = {}
    item_reviews = {}

    for _, row in text_train_df.iterrows():
        user_id, item_id, review = row[0], row[1], row[2]

        if not (isinstance(review, float) and math.isnan(review)):
            if user_id not in user_reviews:
                user_reviews[user_id] = []
            user_reviews[user_id].append(review)

            if item_id not in item_reviews:
                item_reviews[item_id] = []
            item_reviews[item_id].append(review)

    if attack:
        for target_item in []:
            fake_reviews_path = f"data/fake_reviews_{target_item}.npy"
            fake_reviews = np.load(fake_reviews_path, allow_pickle=True).item()

            for user_id, items in fake_reviews.items():
                user_id = str(num_user + int(user_id))
                if user_id not in user_reviews:
                    user_reviews[user_id] = []
                for item_id, review in items.items():
                    if review:
                        user_reviews[user_id].append(review)
                        if item_id not in item_reviews:
                            item_reviews[item_id] = []
                        item_reviews[item_id].append(review)

            tokenizer = BertTokenizer.from_pretrained('bert')
            model = BertModel.from_pretrained('bert')
            model.eval()

            user_embeddings = {}
            for user_id, reviews in user_reviews.items():
                reviews = [review for review in reviews if review]
                combined_reviews = " ".join(reviews)
                embedding = get_bert_embedding([combined_reviews])
                user_embeddings[user_id] = embedding

            all_embeddings = torch.tensor(list(user_embeddings.values()))
            pca = PCA(n_components=100)
            reduced_embeddings = pca.fit_transform(all_embeddings)

            for idx, user_id in enumerate(user_embeddings.keys()):
                user_embeddings[user_id] = reduced_embeddings[idx]

            item_embeddings = {}
            for item_id, reviews in item_reviews.items():
                reviews = [review for review in reviews if review]
                combined_reviews = " ".join(reviews)
                embedding = get_bert_embedding([combined_reviews])
                item_embeddings[item_id] = embedding

            all_embeddings = torch.tensor(list(item_embeddings.values()))
            reduced_embeddings = pca.fit_transform(all_embeddings)

            for idx, item_id in enumerate(item_embeddings.keys()):
                item_embeddings[item_id] = reduced_embeddings[idx]

            with open(f"embedding/user_embeddings_{target_item}.pkl", "wb") as user_file:
                pickle.dump(user_embeddings, user_file)

            with open(f"embedding/item_embeddings_{target_item}.pkl", "wb") as item_file:
                pickle.dump(item_embeddings, item_file)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert')
        model = BertModel.from_pretrained('bert')
        model.eval()

        user_embeddings = {}
        for user_id, reviews in user_reviews.items():
            reviews = [review for review in reviews if review]
            combined_reviews = " ".join(reviews)
            embedding = get_bert_embedding([combined_reviews])
            user_embeddings[user_id] = embedding

        all_embeddings = torch.tensor(list(user_embeddings.values()))
        pca = PCA(n_components=100)
        reduced_embeddings = pca.fit_transform(all_embeddings)

        for idx, user_id in enumerate(user_embeddings.keys()):
            user_embeddings[user_id] = reduced_embeddings[idx]

        item_embeddings = {}
        for item_id, reviews in item_reviews.items():
            reviews = [review for review in reviews if review]
            combined_reviews = " ".join(reviews)
            embedding = get_bert_embedding([combined_reviews])
            item_embeddings[item_id] = embedding

        all_embeddings = torch.tensor(list(item_embeddings.values()))
        reduced_embeddings = pca.fit_transform(all_embeddings)

        for idx, item_id in enumerate(item_embeddings.keys()):
            item_embeddings[item_id] = reduced_embeddings[idx]

        with open("embedding/user_embeddings.pkl", "wb") as user_file:
            pickle.dump(user_embeddings, user_file)

        with open("embedding/item_embeddings.pkl", "wb") as item_file:
            pickle.dump(item_embeddings, item_file)
