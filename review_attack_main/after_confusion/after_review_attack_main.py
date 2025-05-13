import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model_module import RecommendationModel

if __name__ == '__main__':

    dataset = "dataset_name"
    if dataset == "Dataset1":
        num_user = 3441
        attack_num = 172 
        filler_num = 12
        target_item_list = []
    elif dataset == "Dataset2":
        num_user = 2700 
        attack_num = 135 
        filler_num = 10
        target_item_list = []
    else:
        num_user = 880 
        attack_num = 44 
        filler_num = 7
        target_item_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rmse_list = []
    mae_list = []
    for target_id in target_item_list:

        review_attacked = pd.read_csv(f"data/review_{target_id}.csv", header=None)
        review_attacked.columns = ['user_id', 'item_id', 'review', 'rating']

        user_embeddings_dict = pd.read_pickle(f"embeddings/user_embeddings_{target_id}.pkl")
        item_embeddings_dict = pd.read_pickle(f"embeddings/item_embeddings_{target_id}.pkl")

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
            for i in range(0, len(review_attacked), batch_size):
                batch = review_attacked.iloc[i:i + batch_size]

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

            print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(review_attacked)}")

        torch.save(model.state_dict(), f'models/{target_id}_rec_model.pth')

        model.load_state_dict(torch.load(f'models/{target_id}_rec_model.pth'))
        model.eval()

        test_user_vectors = []
        test_item_vectors = []
        test_user_ids = []
        test_item_ids = []
        test_ratings = []

        for _, row in review_attacked.iterrows():
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

        model.eval()
        with torch.no_grad():
            test_predictions = model(test_user_vectors_tensor, test_item_vectors_tensor, test_user_ids_tensor, test_item_ids_tensor).squeeze()

        rmse = np.sqrt(mean_squared_error(test_ratings_tensor.cpu().numpy(), test_predictions.cpu().numpy()))
        mae = mean_absolute_error(test_ratings_tensor.cpu().numpy(), test_predictions.cpu().numpy())

        rmse_list.append(rmse)
        mae_list.append(mae)

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
        user_item_predictions_df.to_csv(f'results/user_item_predictions_{target_id}.csv', index=True)
        print("Done.")

    average_rmse = np.mean(rmse_list)
    average_mae = np.mean(mae_list)

    print(f"Average RMSE: {average_rmse}")
    print(f"Average MAE: {average_mae}")
