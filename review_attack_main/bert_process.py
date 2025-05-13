import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import math
import pickle
from sklearn.decomposition import PCA
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
    dataset = "dataset_name"
    review_data = pd.read_csv(f"data/{dataset}/review.csv", header=None)
    review_data.columns = ['user_id', 'item_id', 'review', 'rating']

    user_reviews = {}
    item_reviews = {}
    for _, row in review_data.iterrows():
        user_id, item_id, review = row[0], row[1], row[2]
        if not (isinstance(review, float) and math.isnan(review)):
            if user_id not in user_reviews:
                user_reviews[user_id] = []
            user_reviews[user_id].append(review)

            if item_id not in item_reviews:
                item_reviews[item_id] = []
            item_reviews[item_id].append(review)

    tokenizer = BertTokenizer.from_pretrained('bert_model_path')
    model = BertModel.from_pretrained('bert_model_path').to(device)
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

    with open(f"embedding/{dataset}/user_embeddings.pkl", "wb") as user_file:
        pickle.dump(user_embeddings, user_file)

    with open(f"embedding/{dataset}/item_embeddings.pkl", "wb") as item_file:
        pickle.dump(item_embeddings, item_file)
