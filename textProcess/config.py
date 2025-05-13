import argparse
import inspect
import torch


class Config:
    device = torch.device("cuda:0")
    train_epochs = 100
    batch_size = 128
    learning_rate = 0.001
    l2_regularization = 1e-6
    learning_rate_decay = 0.99

    word2vec_file = 'path/to/embedding/glove.6B.50d.txt'
    train_file = 'path/to/data/train.csv'
    valid_file = 'path/to/data/valid.csv'
    test_file = 'path/to/data/test.csv'
    model_file = 'path/to/model/best_model.pt'

    review_count = 10
    review_length = 40
    lowest_review_count = 1
    PAD_WORD = '<UNK>'

    kernel_count = 100
    kernel_size = 3
    dropout_prob = 0.5
    cnn_out_dim = 50

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
