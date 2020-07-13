# from base import BaseDataLoader
import torch.utils.data as data
# from base.base_data_loader import BaseDataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import scale


class BaseDataset2(data.Dataset):
    def __init__(self, attribute_feature, label_feature, station='aotizhongxin_aq', mode='train', path='data/single.csv', predict_time_len=8, encoder_time_len=24):
        super(BaseDataset2).__init__()
        print(attribute_feature)
        self.mode = mode
        self.predict_time_len = predict_time_len
        self.encoder_time_len = encoder_time_len
        self.label_feature = label_feature
        # self.target_feature = 'PM25'
        self.target_feature = attribute_feature[0]
        data = pd.read_csv(path)

        self.group_data = data.sort_values(by=["stationId", 'Unnamed: 0'], ascending=True)
        self.group_data = self.group_data.groupby('stationId').get_group(station)

        week_idx = [[0] * 1 for _ in range(self.group_data.shape[0])]
        for i in range(self.group_data.shape[0]):
            week_idx[i] = i // 168 + 1  # 7*24 = 168
        week_idx = pd.DataFrame(week_idx, index=self.group_data.index, columns=['week_idx'])
        group_data = pd.concat([self.group_data, week_idx], axis=1)

        attribute_loc = ['stationId', 'utc_time', 'year', 'month', 'day', 'week_idx']

        data_loc = group_data.loc[:, attribute_loc]
        data = group_data.loc[:, attribute_feature]
        print('PM 25 max. min:', max(data['PM25']), min(data['PM25']))
        print('PM 10 max. min:', max(data['PM10']), min(data['PM10']))

        index_data = data.index
        # data = scale(data.values)
        data_df = pd.DataFrame(data, index=index_data, columns=attribute_feature)
        data_df = data_df.rename(columns={'year': 'norm_year', 'month': 'norm_month', 'day': 'norm_day'})
        full = pd.concat([data_loc, data_df], axis=1)

        test = full.groupby('year').get_group(2018)  # data of 2018 as test
        train_valid = full.groupby('year').get_group(2017)  # last 7 day of one month as valid, data of 2017 and 1<day<23 as train

        train_data = train_valid[train_valid.week_idx % 7 != 0]
        valid_data = train_valid[train_valid.week_idx % 7 == 0]

        self.train_data = train_data
        self.train_label = train_data[label_feature].values
        self.train_target = train_data[self.target_feature].values

        self.valid_data = valid_data

        self.test_data = test
        self.test_label = test[label_feature].values
        self.test_target = test[self.target_feature].values

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)-self.encoder_time_len-self.predict_time_len
        elif self.mode == 'valid':
            return len(self.valid_data)-self.encoder_time_len-self.predict_time_len
        else:
            return len(self.test_data)-self.encoder_time_len-self.predict_time_len

    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.train_data.iloc[:, 6:].values
            label = self.train_label
            feature = data[index: index + self.encoder_time_len, :]
            label_feature = label[index + self.encoder_time_len: index + self.encoder_time_len + self.predict_time_len]
            target = self.train_target[index + self.encoder_time_len: index + self.encoder_time_len + self.predict_time_len]
            return torch.LongTensor(feature), torch.Tensor(label_feature), torch.Tensor(target)

        elif self.mode == 'valid':
            cur_states = self.valid_data.iloc[index:index + self.encoder_time_len]['week_idx']
            first_week_idx = self.valid_data.iloc[index]['week_idx']
            state = (cur_states == first_week_idx)
            not_in_valid = len(state[state == False])

            if not_in_valid > 0:
                cur_data = self.train_data[self.train_data.week_idx == first_week_idx + 1]
                feature_valid = self.valid_data.iloc[index: index + (self.encoder_time_len - not_in_valid), :].iloc[:, 6:].values
                feature_valid = np.concatenate((feature_valid, cur_data.iloc[:not_in_valid:, 6:].values))
                label_valid = cur_data.iloc[not_in_valid:not_in_valid + self.predict_time_len, :][self.label_feature].values
                target_valid = cur_data.iloc[not_in_valid:not_in_valid + self.predict_time_len, :][
                    self.target_feature].values
            else:
                # print(self.valid_data.iloc[index: index + encoder_time_len].values.dtype)
                feature_valid = self.valid_data.iloc[index: index + self.encoder_time_len, :]

                cur_states_2 = self.valid_data.iloc[index + self.encoder_time_len:index + self.encoder_time_len + self.predict_time_len][
                    'week_idx']
                state_2 = (cur_states_2 == first_week_idx)
                not_in_valid_2 = len(state_2[state_2 == False])

                if not_in_valid_2 > 0:
                    cur_data = self.train_data[self.train_data.week_idx == first_week_idx + 1]
                    label_valid = self.valid_data.iloc[index + self.encoder_time_len: index + self.encoder_time_len + (
                            self.predict_time_len - not_in_valid_2), :][self.label_feature]
                    label_valid = np.concatenate((label_valid, cur_data.iloc[:not_in_valid_2, :][self.label_feature].values))

                    target_valid = self.valid_data.iloc[index + self.encoder_time_len: index + self.encoder_time_len + (
                            self.predict_time_len - not_in_valid_2), :][self.target_feature]
                    target_valid = np.concatenate(
                        (target_valid, cur_data.iloc[:not_in_valid_2, :][self.target_feature].values))
                else:
                    label_valid = self.valid_data.iloc[index + self.encoder_time_len:index + self.encoder_time_len + self.predict_time_len, :][self.label_feature].values
                    target_valid = self.valid_data.iloc[index + self.encoder_time_len:index + self.encoder_time_len + self.predict_time_len, :][self.target_feature].values

                feature_valid = feature_valid.iloc[:, 6:].values

            return torch.LongTensor(feature_valid), torch.Tensor(label_valid), torch.Tensor(target_valid)

        else:
            data = self.test_data.iloc[:, 6:].values
            label = self.test_label
            feature = data[index: index + self.encoder_time_len, :]
            label_feature = label[index + self.encoder_time_len: index + self.encoder_time_len + self.predict_time_len]
            target = self.test_target[index + self.encoder_time_len: index + self.encoder_time_len + self.predict_time_len]
            return torch.LongTensor(feature), torch.Tensor(label_feature), torch.Tensor(target)


class dataLoader(DataLoader):
    def __init__(self, attribute_feature, label_feature, station, mode, path, predict_time_len, encoder_time_len, batch_size, num_workeres=1, shuffle=False):
        self.dataset = BaseDataset2(attribute_feature, label_feature, station, mode, path, predict_time_len, encoder_time_len)
        super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workeres)


if __name__ == '__main__':
    valid_loader_C = dataLoader(attribute_feature=["PM25", "PM10"],
                                label_feature=["PM25", "PM10"],
                                station='nongzhanguan_aq',
                                mode="valid",
                                path="/lfs1/users/jbyu/QA_inference/data/single.csv",
                                predict_time_len=1,
                                encoder_time_len=24,
                                batch_size=128,
                                num_workeres=1,
                                shuffle=False)

    for data, target, _ in valid_loader_C:
        # if batch_idx < 3:
        print(data.shape, target.shape)



