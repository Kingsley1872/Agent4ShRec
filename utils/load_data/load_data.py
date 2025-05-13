import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class DataLoader:
    def __init__(self, train_path, test_path, header=None, sep='\t', threshold=4, verbose=True):
        self.train_path = train_path
        self.test_path = test_path
        self.header = header if header is not None else ['user_id', 'item_id', 'rating']
        self.sep = sep
        self.threshold = threshold
        self.verbose = verbose

        self._load_data()

    def _load_data(self):
        self._load_files()
        self.train_matrix, self.train_matrix_implicit = self._convert_to_matrix(self.train_data)
        self.test_matrix, self.test_matrix_implicit = self._convert_to_matrix(self.test_data)

    def _load_files(self):
        if self.verbose:
            print("Loading train/test data from:", self.train_path)
        self.train_data = pd.read_csv(self.train_path, sep=self.sep, names=self.header, engine='python').loc[
            :, ['user_id', 'item_id', 'rating']]
        self.test_data = pd.read_csv(self.test_path, sep=self.sep, names=self.header, engine='python').loc[
            :, ['user_id', 'item_id', 'rating']]

        self.n_users = len(set(self.test_data.user_id.unique()) | set(self.train_data.user_id.unique()))
        self.n_items = len(set(self.test_data.item_id.unique()) | set(self.train_data.item_id.unique()))

        if self.verbose:
            print("Number of users:", self.n_users, ", Number of items:", self.n_items, flush=True)
            print("Train size:", self.train_data.shape[0], ", Test size:", self.test_data.shape[0], flush=True)

    def _convert_to_matrix(self, data_frame):
        row, col, rating, implicit_rating = [], [], [], []
        for line in data_frame.itertuples():
            uid, iid, r = list(line)[1:]
            implicit_r = 1 if r >= self.threshold else 0

            row.append(uid)
            col.append(iid)
            rating.append(r)
            implicit_rating.append(implicit_r)

        matrix = csr_matrix((rating, (row, col)), shape=(self.n_users, self.n_items))
        matrix_implicit = csr_matrix((implicit_rating, (row, col)), shape=(self.n_users, self.n_items))
        return matrix, matrix_implicit

    def get_global_mean_std(self):
        return self.train_matrix.data.mean(), self.train_matrix.data.std()

    def get_all_mean_std(self):
        if not all(hasattr(self, attr) for attr in ['global_mean', 'global_std', 'item_means', 'item_stds']):
            global_mean, global_std = self.get_global_mean_std()
            item_means, item_stds = [global_mean] * self.n_items, [global_std] * self.n_items
            train_matrix_t = self.train_matrix.transpose()
            for iid in range(self.n_items):
                item_vec = train_matrix_t.getrow(iid).toarray()[0]
                ratings = item_vec[np.nonzero(item_vec)]
                if len(ratings) > 0:
                    item_means[iid], item_stds[iid] = ratings.mean(), ratings.std()
            self.global_mean, self.global_std, self.item_means, self.item_stds = global_mean, global_std, item_means, item_stds
        return self.global_mean, self.global_std, self.item_means, self.item_stds

    def get_item_popularity(self):
        item_pops_dict = dict(self.train_data.groupby('item_id').size())
        item_pops = [0] * self.n_items
        for iid in item_pops_dict.keys():
            item_pops[iid] = item_pops_dict[iid]
        return item_pops

    def get_user_nonrated_items(self):
        non_rated_indicator = self.train_matrix.toarray()
        non_rated_indicator[non_rated_indicator > 0] = 1
        non_rated_indicator = 1 - non_rated_indicator
        user_nonrated_items = {}
        for uid in range(self.n_users):
            user_nonrated_items[uid] = list(non_rated_indicator[uid].nonzero()[0])
        return user_nonrated_items

    def get_item_nonrated_users(self, item_id):
        item_vec = np.squeeze(self.train_matrix[:, item_id].toarray())
        item_vec[item_vec > 0] = 1
        non_rated_indicator = 1 - item_vec
        return list(non_rated_indicator.nonzero()[0])
