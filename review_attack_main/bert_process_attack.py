import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import math
import pickle
from sklearn.decomposition import PCA
import os
from tqdm import tqdm

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_bert_embedding(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()

if __name__ == '__main__':
    dataset = "DatasetName"
    ratio = 0.10
    if dataset == "DatasetA":
        target_id_list = ["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9", "ID10"]
    elif dataset == "DatasetB":
        target_id_list = ["ID11", "ID12", "ID13", "ID14", "ID15", "ID16", "ID17", "ID18", "ID19", "ID20"]
    else:
        target_id_list = ["ID21", "ID22", "ID23", "ID24", "ID25", "ID26", "ID27", "ID28", "ID29", "ID30"]

    for target_id in target_id_list:
        review_attack = pd.read_csv(f"data/review_{ratio}/{dataset}/review_{target_id}.csv", header=None)
        review_attack.columns = ['user_id', 'item_id', 'review', 'rating']

        user_reviews = {}
        item_reviews = {}
        for _, row in review_attack.iterrows():
            user_id, item_id, review = row[0], row[1], row[2]
            if not (isinstance(review, float) and math.isnan(review)):
                if user_id not in user_reviews:
                    user_reviews[user_id] = []
                user_reviews[user_id].append(review)

                if item_id not in item_reviews:
                    item_reviews[item_id] = []
                item_reviews[item_id].append(review)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        model.eval()

        user_embeddings = {}
        for user_id, reviews in tqdm(user_reviews.items(), desc="Processing user reviews", leave=True):
            reviews = [review for review in reviews if review]
            embeddings = []
            for review in reviews:
                embedding = get_bert_embedding([review])
                embeddings.append(embedding)
            user_embeddings[user_id] = torch.tensor(embeddings).mean(dim=0).numpy()

        all_embeddings = torch.tensor(list(user_embeddings.values())).to(device)
        pca = PCA(n_components=100)
        reduced_embeddings = pca.fit_transform(all_embeddings.cpu())

        for idx, user_id in enumerate(user_embeddings.keys()):
            user_embeddings[user_id] = reduced_embeddings[idx]

        item_embeddings = {}
        for item_id, reviews in tqdm(item_reviews.items(), desc="Processing item reviews", leave=True):
            reviews = [review for review in reviews if review]
            embeddings = []
            for review in reviews:
                embedding = get_bert_embedding([review])
                embeddings.append(embedding)
            item_embeddings[item_id] = torch.tensor(embeddings).mean(dim=0).numpy()

        all_embeddings = torch.tensor(list(item_embeddings.values())).to(device)
        reduced_embeddings = pca.fit_transform(all_embeddings.cpu())

        for idx, item_id in enumerate(item_embeddings.keys()):
            item_embeddings[item_id] = reduced_embeddings[idx]

        os.makedirs(f"embeddings_{ratio}/{dataset}", exist_ok=True)
        with open(f"embeddings_{ratio}/{dataset}/user_embeddings_{target_id}.pkl", "wb") as user_file:
            pickle.dump(user_embeddings, user_file)

        with open(f"embeddings_{ratio}/{dataset}/item_embeddings_{target_id}.pkl", "wb") as item_file:
            pickle.dump(item_embeddings, item_file)
