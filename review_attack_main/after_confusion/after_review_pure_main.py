import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch.nn as nn
import torch.optim as optim

class RecommendationModel(nn.Module):
    def __init__(self, user_embedding_dim, item_embedding_dim, num_users, num_items, hidden_dim):
        super(RecommendationModel, self).__init__()
        
        self.user_fc = nn.Sequential(
            nn.Linear(user_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.item_fc = nn.Sequential(
            nn.Linear(item_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.user_id_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_id_embedding = nn.Embedding(num_items, hidden_dim)
        
        self.interaction_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_embedding, item_embedding, user_ids, item_ids, embedding_weight=0.5):
        user_vector = self.user_fc(user_embedding)
        item_vector = self.item_fc(item_embedding)
        embedding_interaction = torch.mul(user_vector, item_vector)
        embedding_score = self.interaction_fc(embedding_interaction)

        user_id_vector = self.user_id_embedding(user_ids)
        item_id_vector = self.item_id_embedding(item_ids)
        id_interaction = torch.mul(user_id_vector, item_id_vector)
        id_score = self.interaction_fc(id_interaction)

        final_score = embedding_weight * embedding_score + (1-embedding_weight) * id_score
        return final_score

if __name__ == '__main__':
    dataset = "dataset_name"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    review_data = pd.read_csv(f"data/{dataset}/review.csv", header=None)
    review_data.columns = ['user_id', 'item_id', 'review', 'rating']
    
    user_embeddings_dict = pd.read_pickle(f"embedding/{dataset}/user_embeddings.pkl")
    item_embeddings_dict = pd.read_pickle(f"embedding/{dataset}/item_embeddings.pkl")

    user_embeddings = user_embeddings_dict
    item_embeddings = item_embeddings_dict

    user_embeddings_tensor = torch.tensor(list(user_embeddings.values()), dtype=torch.float32)
    item_embeddings_tensor = torch.tensor(list(item_embeddings.values()), dtype=torch.float32)

    user_embedding_dim = user_embeddings_tensor.shape[1]
    item_embedding_dim = item_embeddings_tensor.shape[1]
    num_users = len(user_embeddings)
    num_items = len(item_embeddings)
    hidden_dim = 128
    model = RecommendationModel(user_embedding_dim, item_embedding_dim, num_users, num_items, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 64
    num_epochs = 20

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, len(review_data), batch_size):
            batch = review_data.iloc[i:i + batch_size]
            
            user_vectors = []
            item_vectors = []
            user_ids = []
            item_ids = []
            ratings = []

            for _, row in batch.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                rating = row['rating']

                user_vectors.append(user_embeddings[user_id])
                item_vectors.append(item_embeddings[item_id])
                user_ids.append(user_id)
                item_ids.append(item_id)
                ratings.append(rating)

            user_vectors_tensor = torch.tensor(user_vectors, dtype=torch.float32).to(device)
            item_vectors_tensor = torch.tensor(item_vectors, dtype=torch.float32).to(device)
            user_ids_tensor = torch.tensor(user_ids, dtype=torch.long).to(device)
            item_ids_tensor = torch.tensor(item_ids, dtype=torch.long).to(device)
            ratings_tensor = torch.tensor(ratings, dtype=torch.float32).to(device)

            predictions = model(user_vectors_tensor, item_vectors_tensor, user_ids_tensor, item_ids_tensor).squeeze()
            loss = criterion(predictions, ratings_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(review_data)}")

    torch.save(model.state_dict(), f'model/{dataset}/rec_model.pth')

    model.load_state_dict(torch.load(f'model/{dataset}/rec_model.pth'))
    model.eval()

    test_user_vectors = []
    test_item_vectors = []
    test_user_ids = []
    test_item_ids = []
    test_ratings = []

    for _, row in review_data.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        rating = row['rating']

        test_user_vectors.append(user_embeddings[user_id])
        test_item_vectors.append(item_embeddings[item_id])
        test_user_ids.append(user_id)
        test_item_ids.append(item_id)
        test_ratings.append(rating)

    test_user_vectors_tensor = torch.tensor(test_user_vectors, dtype=torch.float32).to(device)
    test_item_vectors_tensor = torch.tensor(test_item_vectors, dtype=torch.float32).to(device)
    test_user_ids_tensor = torch.tensor(test_user_ids, dtype=torch.long).to(device)
    test_item_ids_tensor = torch.tensor(test_item_ids, dtype=torch.long).to(device)
    test_ratings_tensor = torch.tensor(test_ratings, dtype=torch.float32).to(device)

    with torch.no_grad():
        test_predictions = model(test_user_vectors_tensor, test_item_vectors_tensor, test_user_ids_tensor, test_item_ids_tensor).squeeze()

    rmse = np.sqrt(mean_squared_error(test_ratings_tensor.cpu().numpy(), test_predictions.cpu().numpy()))
    mae = mean_absolute_error(test_ratings_tensor.cpu().numpy(), test_predictions.cpu().numpy())

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    user_item_predictions = {}

    with torch.no_grad():
        user_ids = list(user_embeddings.keys())
        item_ids = list(item_embeddings.keys())

        item_vectors_tensor = torch.tensor(list(item_embeddings.values()), dtype=torch.float32).to(device)
        item_ids_tensor = torch.tensor(item_ids, dtype=torch.long).to(device)

        for user_id in user_ids:
            user_vector = torch.tensor(user_embeddings[user_id], dtype=torch.float32).unsqueeze(0).to(device)
            user_id_tensor = torch.tensor([user_id], dtype=torch.long).to(device)

            user_vector_batch = user_vector.repeat(len(item_ids), 1)
            user_id_tensor_batch = user_id_tensor.repeat(len(item_ids))

            scores = model(user_vector_batch, item_vectors_tensor, user_id_tensor_batch, item_ids_tensor).squeeze()
            user_item_predictions[user_id] = scores.tolist()

    user_item_predictions_df = pd.DataFrame.from_dict(user_item_predictions, orient='index')
    user_item_predictions_df.columns = [f'item_{i}' for i in range(len(item_embeddings))]
    user_item_predictions_df.to_csv(f'result/{dataset}/user_item_predictions.csv', index=True)
    print("User-item predictions saved to user_item_predictions.csv")
