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

        self.mode = mode

        feature_path = '/lfs1/users/jbyu/QA_baseline/np_data/{}_Pm25_Pm10_feature_hour{}.npy'.format(mode, time_len)
        target_path = '/lfs1/users/jbyu/QA_baseline/np_data/{}_Pm25_Pm10_target_hour{}.npy'.format(mode, time_len)
        self.embed_path = embed_path
        self.feature = np.load(feature_path)
        self.target = np.load(target_path)
        np.random.seed(0)
        np.random.shuffle(self.feature)
        np.random.shuffle(self.target)

        print(self.feature.shape)

    def __len__(self):
        return self.feature.shape[0]


    # finetune number embedding
    def __getitem__(self, idx):

        pm25_feature = torch.LongTensor(self.feature[idx][:, 0])
        pm25_target = self.target[idx][:, 0]
        pm10_feature = torch.LongTensor(self.feature[idx][:, 1])
        pm10_target = self.target[idx][:, 1]
        return pm25_feature, pm10_feature, torch.LongTensor(pm25_target), torch.LongTensor(pm10_target)


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
