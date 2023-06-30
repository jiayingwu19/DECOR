import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from Process.load_graph import *
from tqdm import tqdm
from torch_geometric.nn import MLP
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str)
parser.add_argument('--model_name', default='gin_decor', type=str)
parser.add_argument('--u_thres', default=3, type=int)
parser.add_argument('--iters', default=20, type=int)

args = parser.parse_args()
## if multi-gpu is needed for training on large social graphs, uncomment the commented codes and run the following command
## CUDA_VISIBLE_DEVICES=0,1,2,3 python src/gcn_decor.py --dataset_name [dataset_name] 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

device = torch.device("cuda")


class GINDecor(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, n_classes, mlp_width):
        super().__init__()

        self.adj_transform = MLP([3, mlp_width, 2])
        # self.adj_transform = torch.nn.DataParallel(MLP([3, mlp_width, 2]), device_ids=[0, 1, 2, 3])
        self.softmax = nn.Softmax(dim = -1)
        self.resid_weight = torch.nn.Parameter(torch.Tensor([0.]))
        self.nn1 = MLP([in_dim, hid_dim])
        self.nn2 = MLP([hid_dim, n_classes])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()


    def forward(self, data):
        indices = torch.nonzero(data.adj.view(-1, 1), as_tuple=True)[0].cpu().detach().numpy()
        adj_flattened = torch.zeros(data.adj.view(-1, 1).shape[0]).to(data.adj.device)

        x, adj = data.x, torch.cat((data.adj.view(-1, 1)[indices], data.xdeg.view(-1, 1)[indices], data.ydeg.view(-1, 1)[indices]), 1)
        adj_mask = self.softmax(self.adj_transform(adj))[:, 1]
        adj_flattened[indices] = adj_mask
        adj_mask = adj_flattened.reshape(data.adj.shape[0], data.adj.shape[1])
        adj = data.adj * adj_mask
        adj = adj + torch.eye(*adj.shape).to(data.adj.device)


        rowsum = torch.sum(adj, dim = 1)
        D_row = torch.pow(rowsum, -0.5).flatten()
        D_row[torch.isinf(D_row)] = 0.
        D_row = torch.diag(D_row)
        colsum = torch.sum(adj, dim = 0)
        D_col = torch.pow(colsum, -0.5).flatten()
        D_col[torch.isinf(D_col)] = 0.
        D_col = torch.diag(D_col)
        adj = adj.mm(D_col).transpose(0, 1).mm(D_row).transpose(0, 1) 
        

        output = torch.mm(adj, x)
        output = self.resid_weight * x + output
        hid = self.nn1(output)
        hid = self.dropout(hid)
        output = torch.mm(adj, hid)
        output = self.resid_weight * hid + output
        output = self.nn2(output)   


        return output


def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def train(data, iter):

    model = GINDecor(768, 64, 2, mlp_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    model.train()

    for epoch in tqdm(range(800)):
        optimizer.zero_grad()
        out = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out[data.train_mask], data.y_train)
        _, pred = out[data.train_mask].max(dim = -1)
        train_acc = pred.eq(data.y_train).sum().item() / len(data.y_train)
        _, testpred = out[data.test_mask].max(dim = -1)
        test_acc = testpred.eq(data.y_test).sum().item() / len(data.y_test)
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    y_pred = pred[data.test_mask]
    acc = accuracy_score(data.y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    precision, recall, fscore, _ = score(data.y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')


    print(['Global Test Accuracy:{:.4f}'.format(acc),
        'Precision:{:.4f}'.format(precision),
        'Recall:{:.4f}'.format(recall),
        'F1:{:.4f}'.format(fscore)])
    print("-----------------End of Iter {:03d}-----------------".format(iter))

    
    return acc, precision, recall, fscore


datasetname = args.dataset_name
u_thres = args.u_thres
iterations = args.iters
mlp_width = 16 if datasetname == 'politifact' else 8
data = load_graph_decor(datasetname, u_thres).to(device)

test_accs = []
prec_all, rec_all, f1_all = [], [], []

for iter in range(iterations):
    set_seed(iter)
    acc, prec, recall, f1 = train(data, iter)
    test_accs.append(acc)
    prec_all.append(prec)
    rec_all.append(recall)
    f1_all.append(f1)

print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs) / iterations, sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))


with open('logs/log_' +  datasetname + '_train80pct' + '_' + args.model_name + '_user_t' + str(u_thres) + '.iter' + str(iterations), 'a+') as f:
    f.write('All Acc.s:{}\n'.format(test_accs))
    f.write('All Prec.s:{}\n'.format(prec_all))
    f.write('All Rec.s:{}\n'.format(rec_all))
    f.write('All F1.s:{}\n'.format(f1_all))
    f.write('Average acc.: {} \n'.format(sum(test_accs) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))
    f.write('\n')


