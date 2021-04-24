from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import time


class LaraDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv, user_emb_matrix):
        self.train_csv = pd.read_csv(train_csv, header=None)
        # 使用dataset时，最好先进行loc定位之后，在getItem中直接查找，查找时间会下降3个数量级
        self.user = self.train_csv.loc[:, 0]
        self.item = self.train_csv.loc[:, 1]
        self.attr = self.train_csv.loc[:, 2]
        self.user_emb_matrix = pd.read_csv(user_emb_matrix, header=None)
        self.user_emb_values = np.array(self.user_emb_matrix[:])

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        user_emb = self.user_emb_values[user]
        # 处理属性，将字符串类型转换为整数
        attr = self.attr[idx][1:-1].split()
        attr = torch.tensor(list([int(item) for item in attr]), dtype=torch.long)
        attr = np.array(attr)
        end = time.time()
        return user, item, attr, user_emb


'''
dataSet = LaraDataset('data/neg_data.csv', 'data/user_emb.csv')
dataIter = torch.utils.data.DataLoader(dataSet, batch_size=2, shuffle=False)
iter = dataIter.__iter__()
u,i,a,ue = iter.next()
print(u,i,a,ue)
'''
'''
for user, item, attr, user_emb in dataIter:
    print(user)
    print(item)
    print(attr)
    print(user_emb)
    break
'''