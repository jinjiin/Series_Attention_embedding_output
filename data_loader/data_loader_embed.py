import torch.utils.data as data
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader
import torch


class BaseDataset(data.Dataset):
    def __init__(self, mode, time_len, embed_path="embed_128_no_l2.npy"):
        super(BaseDataset).__init__()

        # mode:train/valid
        # time_len:6/12/24/48
        # embed_path：embed_{}_no_l2.npy：50\64\128\256\512，no_l2\l2
        # feature_choice:pm25/pm10

        pm25_feature_path = '/lfs1/users/jbyu/QA_baseline/np_data/{}_hour{}_pm25_feature_embed50.npy'\
            .format(mode, time_len)
        pm25_target_path = '/lfs1/users/jbyu/QA_baseline/np_data/{}_hour{}_pm25_target_embed50.npy'\
            .format(mode, time_len)
        pm10_feature_path = '/lfs1/users/jbyu/QA_baseline/np_data/{}_hour{}_pm10_feature_embed50.npy'\
            .format(mode,time_len)
        pm10_target_path = '/lfs1/users/jbyu/QA_baseline/np_data/{}_hour{}_pm10_target_embed50.npy'.\
            format(mode, time_len)

        self.pm25_feature = torch.FloatTensor(np.load(pm25_feature_path))
        self.pm25_target = torch.FloatTensor(np.load(pm25_target_path))
        self.pm10_feature = torch.FloatTensor(np.load(pm10_feature_path))
        self.pm10_target = torch.FloatTensor(np.load(pm10_target_path))

        self.pm25_feature_mean = torch.mean(self.pm25_feature, dim=0)
        self.pm25_feature_std = torch.std(self.pm25_feature, dim=0)
        self.pm10_feature_mean = torch.mean(self.pm10_feature, dim=0)
        self.pm10_feature_std = torch.std(self.pm10_feature, dim=0)

        self.pm25_feature = (self.pm25_feature - self.pm25_feature_mean) / self.pm25_feature_std
        self.pm10_feature = (self.pm10_feature - self.pm10_feature_mean) / self.pm10_feature_std

        print(self.pm25_feature.shape)

    def __len__(self):
        return self.pm25_feature.shape[0]

    # fixed number embedding
    def __getitem__(self, idx):
        pm25_feature = self.pm25_feature[idx]
        pm10_feature = self.pm10_feature[idx]
        pm25_target = self.pm25_target[idx]
        pm10_target = self.pm10_target[idx]
        return pm25_feature, pm10_feature, pm25_target, pm10_target

    # finetune number embedding
    # def __getitem__(self, idx):
    #
    #     pm25_feature = torch.LongTensor(self.feature[idx][:, 0])
    #     pm25_target = self.target[idx][:, 0]
    #     pm10_feature = torch.LongTensor(self.feature[idx][:, 1])
    #     pm10_target = self.target[idx][:, 1]
    #     return pm25_feature, pm10_feature, torch.FloatTensor(pm25_target), torch.FloatTensor(pm10_target)


class dataLoader(DataLoader):
    def __init__(self, mode, time_len, embed_path, batch_size, num_workeres=1, shuffle=True):
        self.dataset = BaseDataset(mode, time_len, embed_path)
        super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workeres)


if __name__ == '__main__':
    valid_loader = dataLoader(
                                mode='train',
                                time_len=12,
                                embed_path="embed_128_no_l2.npy",
                                batch_size=512,
                                num_workeres=1,
                                shuffle=True)

    for batch_idx, (pm25, target) in enumerate(valid_loader):
        print(pm25.shape, target.shape)
