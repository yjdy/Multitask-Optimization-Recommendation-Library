import numpy as np
import pandas as pd
import torch
import pickle
import os


class MTDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]


class AliExpressDataset(MTDataset):
    """
    AliExpress Dataset
    This is a dataset gathered from real-world traffic logs of the search system in AliExpress
    Reference:
        https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
        Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
    """

    def __init__(self, dataset_path):
        if dataset_path.endswith("csv"):
            self.categorical_data, self.numerical_data, self.labels, \
                self.numerical_num, self.field_dims = load_data(dataset_path)

        elif dataset_path.endswith('pkl'):
            with open(dataset_path, 'rb') as fp:
                data = pickle.load(fp)
            self.categorical_data = data["categorical_data"]
            self.numerical_data = data["numerical_data"]
            self.labels = data["labels"]
            self.numerical_num = self.numerical_data.shape[1]
            self.field_dims = np.max(self.categorical_data, axis=0) + 1
        else:
            raise NotImplementedError


class AliExpressDatasetV2(torch.utils.data.Dataset):
    def __init__(self, data_df: pd.DataFrame):

        self.categorical_data, self.numerical_data, self.labels, \
            self.numerical_num,self.field_dims = load_data(data_df)


from sklearn.model_selection import train_test_split


def split_dataset(path, ratio=0.5):
    data = pd.read_csv(path)
    test, valid = train_test_split(data, test_size=ratio)
    valid = AliExpressDatasetV2(valid)
    test = AliExpressDatasetV2(test)

    return valid, test


def load_data(data):
    if isinstance(data,str):
        data = pd.read_csv(data).to_numpy()[:, 1:]
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()[:, 1:]
    categorical_data = data[:, :16].astype(np.int32)
    numerical_data = data[:, 16: -2].astype(np.float32)
    labels = data[:, -2:].astype(np.float32)
    numerical_num = numerical_data.shape[1]
    field_dims = np.max(categorical_data, axis=0) + 1
    return categorical_data, numerical_data, labels, numerical_num, field_dims


def create_pikle_files(dataset_path, mode='train'):
    categorical_data, numerical_data, labels, _, _ = load_data(dataset_path)

    features = {
        "categorical_data": categorical_data,
        "numerical_data": numerical_data,
        "labels": labels
    }

    with open(f'{os.path.dirname(dataset_path)}/{mode}.pkl', 'wb') as fp:
        pickle.dump(features, fp)
