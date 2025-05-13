import os
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from model import DeepCoNN
from utils import load_embedding, DeepCoNNDataset, predict_mse, date, DeepCoNNDatasetTest, predict_mse_test
import pickle
import pandas as pd


def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse = predict_mse(model, train_dataloader, config.device)
    valid_mse = predict_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        for batch in train_dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews)
            loss = F.mse_loss(predict, ratings, reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_samples += len(predict)

        lr_sch.step()
        model.eval()
        valid_mse = predict_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model, target_id=''):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_loss, all_pred = predict_mse_test(model, dataloader, config.device)
    result_dir = os.path.join("result")
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"output{target_id}.csv")
    
    user_ids = range(2700)
    item_ids = range(3090)
    data = []
    for user_id in user_ids:
        for item_id in item_ids:
            index = user_id * len(item_ids) + item_id
            pred = all_pred[index]
            data.append([user_id, item_id, pred])
    
    df = pd.DataFrame(data, columns=["userid", "itemid", "pred"])
    df.to_csv(result_file, index=False)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


if __name__ == '__main__':
    config = Config()
    print(config)
    print(f'{date()}## Load embedding and data...')
    for target_id in []:
        word_emb, word_dict = load_embedding(config.word2vec_file)
        config.model_file = f'model/best_model.pt'
        config.train_file = f'data/train.csv'
        config.test_file = f'data/test.csv'
        train_dataset = DeepCoNNDataset(config.train_file, word_dict, config)
        valid_dataset = DeepCoNNDataset(config.valid_file, word_dict, config, retain_rui=False)
        test_dataset = DeepCoNNDatasetTest(config.test_file, word_dict, config, retain_rui=True)
        train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
        test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)

        model = DeepCoNN(config, word_emb).to(config.device)
        del train_dataset, valid_dataset, test_dataset, word_emb, word_dict

        os.makedirs(os.path.dirname(config.model_file), exist_ok=True)
        train(train_dlr, valid_dlr, model, config, config.model_file)
        test(test_dlr, torch.load(config.model_file), target_id)
