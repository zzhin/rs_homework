import torch
from torch import nn
import time
import torch.utils.data
from tqdm import tqdm
import support
import test


# 超参数设置
alpha = 0  # 正则项参数
attr_num = 18
attr_present_dim = 5
batch_size = 1024
hidden_dim = 100
user_emb_dim = attr_num
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
epoch = 400


def param_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.xavier_normal_(m.bias.unsqueeze(0))
    else:
        pass


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_attr_matrix = nn.Embedding(2*attr_num, attr_present_dim)
        self.l1 = nn.Linear(attr_num*attr_present_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.h = nn.Tanh()
        self.__init_param__()

    def __init_param__(self):
        for md in self.G_attr_matrix.modules():
            torch.nn.init.xavier_normal_(md.weight)
        for md in self.modules():
            param_init(md)

    def forward(self, attribute_id):
        attr_present = self.G_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num*attr_present_dim])
        o1 = self.h(self.l1(attr_feature))
        o2 = self.h(self.l2(o1))
        o3 = self.h(self.l3(o2))
        return o3


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_attr_matrix = nn.Embedding(2*attr_num, attr_present_dim)
        self.l1 = nn.Linear(attr_num*attr_present_dim+user_emb_dim, hidden_dim, bias=True)
        self.h = nn.Tanh()
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.__init_param__()

    def __init_param__(self):
        for md in self.D_attr_matrix.modules():
            torch.nn.init.xavier_normal_(md.weight)
        for md in self.modules():
            param_init(md)

    def forward(self, attribute_id, user_emb):
        attribute_id = attribute_id.long()
        attr_present = self.D_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num*attr_present_dim])
        emb = torch.cat((attr_feature, user_emb), 1)
        emb = emb.float()
        o1 = self.h(self.l1(emb))
        o2 = self.h(self.l2(o1))
        d_logit = self.l3(o2)
        d_prob = torch.sigmoid(d_logit)
        return d_prob, d_logit


def train(g, d, train_loader, neg_loader, epochs, g_optim, d_optim, datasetLen):
    g = g.to(device)
    d = d.to(device)
    print("train on ", device)
    # loss 函数，BCELoss是二分类的交叉熵，需要的是经过sigmoid层的数据, 范围要求的是[0,1]之间的数
    loss = torch.nn.BCELoss()
    start = time.time()
    for i_epo in tqdm(range(epochs)):
        i = 0
        neg_iter = neg_loader.__iter__()
        # 训练D
        d_loss_sum = 0.0
        for user, item, attr, user_emb in train_loader:
            if i*batch_size >= datasetLen:
                break
            # 取出负采样的样本
            _, _, neg_attr, neg_user_emb = neg_iter.next()
            neg_attr = neg_attr.to(device)
            neg_user_emb = neg_user_emb.to(device)
            attr = attr.to(device)
            user_emb = user_emb.to(device)
            fake_user_emb = g(attr)  # 根据item的属性生成用户表达
            d_real, d_logit_real = d(attr, user_emb)
            d_fake, d_logit_fake = d(attr, fake_user_emb)
            d_neg, d_logit_neg = d(neg_attr, neg_user_emb)
            # d_loss分成三部分, 正样本，生成的样本，负样本
            d_optim.zero_grad()
            d_loss_real = loss(d_real, torch.ones_like(d_real))
            d_loss_fake = loss(d_fake, torch.zeros_like(d_fake))
            d_loss_neg = loss(d_neg, torch.zeros_like(d_neg))
            d_loss_sum = torch.mean(d_loss_real + d_loss_fake+d_loss_neg)
            d_loss_sum.backward()
            d_optim.step()
            i += 1
        # 训练G
        g_loss = 0.0
        for user, item, attr, user_emb in train_loader:
            # g loss
            g_optim.zero_grad()
            attr = attr.long()
            attr = attr.to(device)
            fake_user_emb = g(attr)
            fake_user_emb.to(device)
            d_fake, _ = d(attr, fake_user_emb)
            g_loss = loss(d_fake, torch.ones_like(d_fake))
            g_loss.backward()
            g_optim.step()
        end = time.time()
        print("\n time:%.3fs, d_loss:%.4f, g_loss:%.4f " % ((end - start), d_loss_sum, g_loss))
        start = end
        # test---
        item, attr = test.get_test_data()
        item = item.to(device)
        attr = attr.to(device)
        item_user = g(attr)
        test.to_valuate(item, item_user)
        g_optim.zero_grad()  # 生成器清零梯度
        # 保存模型
        if i_epo % 10 == 0:
            print("model has been saved! locate in data/result/")
            torch.save(g.state_dict(), 'data/result/g_'+str(i_epo)+".pt")
            torch.save(d.state_dict(), 'data/result/d_' + str(i_epo) + ".pt")


def run():
    train_dataset = support.LaraDataset('data/train/train_data.csv', 'data/train/user_emb.csv')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    neg_dataset = support.LaraDataset('data/train/neg_data.csv', 'data/train/user_emb.csv')
    neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    generator = Generator()
    discriminator = Discriminator()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=alpha)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=alpha)
    # 因为负样本的数据量要小一些，为了训练方便，使用负样本的长度来训练
    train(generator, discriminator, train_loader, neg_loader, epoch, g_optimizer, d_optimizer, neg_dataset.__len__())
