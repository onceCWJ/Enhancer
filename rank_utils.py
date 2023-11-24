import numpy as np
import scipy.sparse as sp
from random import shuffle
import torch
from tqdm import tqdm
import os
import pickle as pkl
import pandas as pd

def load_knowledge(path):
    with open(path, 'rb') as f:
        emb = pkl.load(f) # [1026, 64]
        return emb


def load_sample(path):
    with open(path, 'rb') as f:
        train_data = pkl.load(f) # [stock_news, mask, stock_prices, stock_return]...
        train_market, train_price, train_label, train_base, stock_mask = [], [], [], [], []
        for i in train_data:
            [one_market, one_price, one_label, one_base, one_stock_mask] = i
            train_market.append(np.array(one_market))
            train_price.append(np.array(one_price))
            train_label.append(np.array(one_label))
            train_base.append(np.array(one_base))
            stock_mask.append(np.array(one_stock_mask))
        return np.array(train_market), np.array(train_price), np.array(train_label), np.array(train_base), np.array(stock_mask)


def load_stocklist(path):
    if 'Ashare' in path:
        stock_pd = pd.read_csv(path.split('code2ID.txt')[0]+'ticket_info.csv')
        ticket_list = stock_pd['code'].tolist()
        ticket_list = [(6-len(str(ticket)))*'0'+str(ticket) for ticket in ticket_list]
        return ticket_list
    else:
        with open(path) as f:
            lines = f.readlines()
        return [s.strip() for s in lines]


def load_date(path):
    with open(path, 'rb') as f:
        test_date = pkl.load(f) # [stock_news, mask, stock_prices, stock_return]...
        return test_date


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if len(sparse_mx.nonzero()[0]) == 0:
        # 空矩阵
        r, c = sparse_mx.shape
        return torch.sparse.FloatTensor(r, c)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_tensor_to_sparse(dense_mx):
    return sparse_mx_to_torch_sparse_tensor( sp.coo.coo_matrix(dense_mx) )


def makedirs(dirs: list):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    return

# def change_adj_2dense(adj_list):
#     ans = []
#     for i in adj_list:
#         cache = []
#         for j in i:
#             cache.append(j.to_dense()) # sparse tensor转换回稠密矩阵
#         ans.append(torch.cat(cache, dim=1))
#     adj_dense = torch.cat(ans, dim=0)
#     adj_shape = [i.shape[1] for i in adj_list[0]]
#     return adj_dense, adj_shape

def change_adj_2dense(adj_list, news_nodeID):
    ans = []
    N = len(adj_list)
    for i in range(N):
        cache = []
        for j in range(N):
            if i==0 and j==0:
                cache.append(torch.eye(len(news_nodeID)))
                continue
            elif i==0:
                cache.append(adj_list[i][j].to_dense()[news_nodeID]) # sparse tensor转换回稠密矩阵
            elif j==0:
                cache.append(adj_list[i][j].to_dense().T[news_nodeID].T)
            else:
                cache.append(adj_list[i][j].to_dense())

        ans.append(torch.cat(cache, dim=1))
    adj_shape = [i.shape[0] for i in ans]
    adj_dense = torch.cat(ans, dim=0)
    return adj_dense, adj_shape