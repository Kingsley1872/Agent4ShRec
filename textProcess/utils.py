import time
import pandas as pd
import torch
from torch.utils.data import Dataset


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict


def predict_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(device), batch)
            predict = model(user_reviews, item_reviews)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count


def predict_mse_test(model, dataloader, device):
    mse, sample_count = 0, 0
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, ratings = map(lambda x: x.to(device), batch)
            predict = model(user_reviews, item_reviews)
            all_preds.extend(predict.cpu().view(-1).numpy())
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count, all_preds


class CustomDataset(Dataset):
    def __init__(self, data_path, word_dict, config, retain_rui=True):
        self.word_dict = word_dict
        self.config = config
        self.retain_rui = retain_rui
        self.PAD_WORD_idx = self.word_dict[config.PAD_WORD]
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.lowest_r_count = config.lowest_review_count

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)
        user_reviews = self._get_reviews(df)
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

        self.user_reviews = user_reviews
        self.item_reviews = item_reviews
        self.rating = rating

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]
            if self.retain_rui:
                reviews = df_data['review'].to_list()
            else:
                reviews = df_data['review'][df_data[costar] != costar_id].to_list()
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]
        return reviews

    def _review2id(self, review):
        if not isinstance(review, str):
            return []
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])
            else:
                wids.append(self.PAD_WORD_idx)
        return wids


class CustomDatasetTest(Dataset):
    def __init__(self, data_path, word_dict, config, retain_rui=True):
        self.word_dict = word_dict
        self.config = config
        self.retain_rui = retain_rui
        self.PAD_WORD_idx = self.word_dict[config.PAD_WORD]
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.lowest_r_count = config.lowest_review_count

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)
        item_reviews = self._get_reviews_item(df, 'itemID', 'userID')
        user_reviews = self._get_reviews(df)
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

        self.user_reviews = user_reviews
        self.item_reviews = item_reviews
        self.rating = rating

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))
        lead_reviews = []
        for lead_id in range(len(df[lead].unique())):
            df_data = reviews_by_lead[lead_id]
            df_data = df_data[df_data['review'].apply(lambda x: len(x) > 0)]
            reviews = df_data['review'].to_list()
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            num_costars = len(df[costar].unique())
            if self.retain_rui:
                lead_reviews.extend([reviews] * num_costars)
        return torch.LongTensor(lead_reviews)

    def _get_reviews_item(self, df, lead='userID', costar='itemID'):
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))
        lead_reviews_dict = {}
        lead_reviews = []
        for lead_id in range(len(df[lead].unique())):
            if lead_id in lead_reviews_dict:
                continue
            df_data = reviews_by_lead[lead_id]
            df_data = df_data[df_data['review'].apply(lambda x: len(x) > 0)]
            reviews = df_data['review'].to_list()
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews_dict[lead_id] = reviews
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            lead_reviews.append(lead_reviews_dict[lead_id])
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]
        return reviews

    def _review2id(self, review):
        if not isinstance(review, str):
            return []
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])
            else:
                wids.append(self.PAD_WORD_idx)
        return wids
