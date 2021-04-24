import pandas as pd
import torch
from sklearn.utils import shuffle
import numpy as np
import time
'''
@author: zhou zhi hui
@date: 2021-04-09 21:16
@dataSet: movieLen-100k u.data, 80% train, 20% test
@functional:
implementations of 'Probabilistic Matrix Factorization' in PyTorch
@evaluate metric: p@10=0.11
'''
ratings = pd.read_csv('data/u.data', sep='\t')
ratings = shuffle(ratings)
train_ratings = ratings[0: int(len(ratings)*0.8)]
test_ratings = ratings[int(len(ratings)*0.2):]
user = ratings['userId']
item = ratings['movieId']
rating_matrix = torch.zeros(len(set(user)), len(set(item)))
user_map, item_map = {}, {}
for u in user:
    if user_map.get(u) is None:
        user_map[u] = len(user_map)
for i in item:
    if item_map.get(i) is None:
        item_map[i] = len(item_map)
# 构建评分矩阵
for u, i, r in zip(train_ratings['userId'], train_ratings['movieId'], train_ratings['rating']):
    rating_matrix[user_map.get(u)][item_map.get(i)] = r
n_users, n_movies = rating_matrix.shape
print("#user:", n_users, "#movies:", n_movies)
# 对所有评分信息进行归一化，限制范围到[0,1]之间
min_rating, max_rating = ratings['rating'].min(), ratings['rating'].max()
non_zero_idx = torch.nonzero(rating_matrix, as_tuple=True)
rating_matrix[non_zero_idx] = (rating_matrix[non_zero_idx]-min_rating)/(max_rating-min_rating)
# 嵌入维度
latent_dim = 5
# 构建测试集中的user-item, key是user, value是测试集中user交互过的item
test_user_map = {}
for u, i in zip(test_ratings['userId'], test_ratings['movieId']):
    item_list = test_user_map.get(user_map[u])
    if item_list is None:
        i_list = [item_map[i]]
        test_user_map[user_map[u]] = i_list
    else:
        item_list.append(item_map[i])
        test_user_map[user_map[u]] = item_list
count = 0
for key in test_user_map.keys():
    item_len = len(test_user_map[key])
    count += item_len
print("# avg of item rated by user in test set :", count/len(test_user_map))


class PMF(torch.nn.Module):
    def __init__(self, lam_u=0.05, lam_v=0.05):
        super().__init__()
        self.user_features = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, (n_users, latent_dim))))
        self.movie_features = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.1, (n_movies, latent_dim))))
        # lam_u, lam_v是user, item的正则化系数
        self.lam_u = lam_u
        self.lam_v = lam_v
    
    def forward(self, matrix):
        non_zero_mask = (matrix != 0).type(torch.FloatTensor)
        predicted = torch.sigmoid(torch.mm(self.user_features, self.movie_features.t()))
        diff = (matrix - predicted)**2
        prediction_error = torch.sum(diff*non_zero_mask)# 只求非零的评分
        u_regularization = self.lam_u * torch.sum(self.user_features.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(self.movie_features.norm(dim=1))
        return prediction_error + u_regularization + v_regularization


def test(k=10):
    p_k_avg = 0.0
    # 为user推荐item
    for u_tmp in test_user_map.keys():
        user_idx = int(u_tmp)
        # 开始测试
        predictions = torch.mm(pmf.user_features[user_idx, :].view(1, -1), pmf.movie_features.t())
        recommend_item = torch.argsort(-predictions.squeeze())
        # 进行p@k评价指标
        cnt = 0
        for i_tmp in range(k):
            if recommend_item[i_tmp] in test_user_map[u_tmp]:
                cnt += 1
        p_k = cnt/min(k, len(test_user_map[u_tmp]))
        p_k_avg += p_k
    # 打印P@K指标
    print("p@%d %.4f" % (k, p_k_avg/len(test_user_map)))
    return p_k_avg/len(test_user_map)


# train
pmf = PMF(lam_u=0.05, lam_v=0.05)
optimizer = torch.optim.Adam(pmf.parameters(), lr=0.01)

f = open('outputs/result_'+str(time.time_ns())+".csv", 'a')
f.write("epoch,loss,p_10\n")
for epoch in range(300):
    optimizer.zero_grad()
    loss = pmf(rating_matrix)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("epoch:%d, loss:%.4f" % (epoch, loss))
        p_10 = test()
        f.write(str(epoch)+","+str(loss.item())+","+str(p_10)+"\n")
f.close()
