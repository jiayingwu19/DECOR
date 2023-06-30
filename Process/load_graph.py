import pickle 
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data



def load_graph_decor(obj, u_thres = 3):

    news_features = pickle.load(open('data/news_features/' + obj + '_bert_raw_768d.pkl', 'rb'))
    graph_dict = pickle.load(open('data/user_news_graph/weighted/' + obj + '_un_relations_t' + str(u_thres) + '_raw.pkl', 'rb'))
    mask_dict = pickle.load(open('data/temp_splits/' + obj + '_train_test_mask_80_20_temp.pkl', 'rb'))

    train_mask, test_mask = mask_dict['train_mask'], mask_dict['test_mask']
    y_train, y_test = mask_dict['y_train'], mask_dict['y_test']
    A_un = torch.Tensor(graph_dict['A_un'])

    # DECOR thresholds the maximum number of engagement between a certain user and a certain article at 1% of num_news.
    s = round(A_un.shape[1] / 100)
    A_un_new = torch.where(A_un < s, A_un, torch.tensor(s, dtype=A_un.dtype))
    adj = A_un_new.transpose(0, 1).matmul(A_un_new)

    # degrees
    xdeg, ydeg = adj.sum(0), adj.sum(0)
    xdeg = xdeg.view(-1, 1)
    xdeg, ydeg = xdeg.repeat(1, adj.shape[0]), ydeg.repeat(adj.shape[1], 1)

    # co-engagement
    A_un_thres1 = torch.where(A_un < 1, A_un, torch.tensor(1., dtype=A_un.dtype))
    adj_thres1 = A_un_thres1.transpose(0, 1).matmul(A_un_thres1)


    return Data(x = news_features, adj = adj_thres1, xdeg = xdeg, ydeg = ydeg,
                train_mask = train_mask, test_mask = test_mask,
                y_train = y_train, y_test = y_test)
