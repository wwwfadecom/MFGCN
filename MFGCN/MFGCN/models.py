import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import numpy as np
import math
from scipy.sparse import coo_matrix

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x
class GCN1(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc15 = GraphConvolution(nhid, nhid*2)
        # self.gc16 = GraphConvolution(nhid*2, nhid*4)
        self.gc2 = GraphConvolution(nhid*2, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.relu(self.gc15(x, adj))
        # x = F.dropout(x, self.dropout, training = self.training)
        # x = F.relu(self.gc16(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x
class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN2, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc15 = GraphConvolution(nhid, nhid)
        self.gc16 = GraphConvolution(nhid,nhid*4)
        self.gc2 = GraphConvolution(nhid*4, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.relu(self.gc15(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = F.relu(self.gc16(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj.to_dense())
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class MFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout,sadj,fadj):
        super(MFGCN, self).__init__()
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self.estimated_adj1 = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(((sadj+fadj)/2).to_dense())
        self.normalized_adj = self.normalize()
        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.symmetric = False
        self.nclass = nclass
        self.dropout = dropout
        # self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()
        self.w_weight = 0
        # self.MLP = nn.Sequential(
        #     nn.Linear(nhid2, nclass),
        #     nn.LogSoftmax(dim=1)
        # )
        self.m1 = nn.Linear(nhid2, nclass)
        self.m2 = nn.LogSoftmax(dim=1)
        self.cluster_layer = Parameter(torch.Tensor(nclass, nhid2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
    def forward(self, x, sadj, fadj):

        self.estimated_adj1 = Parameter(self.normalized_adj.cuda())

        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        # estimator=coo_matrix(estimator).to
        com1 = self.CGCN(x, self.estimated_adj1)  # Common_GCN out1 -- sadj structure graph
        # com2 = self.CGCN(x, self.estimator)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        Xcom = com1
        ##attention
        # emb = torch.stack([emb1, emb2, Xcom], dim=1)
        # emb, att = self.attention(emb)
        att=0
        emb = ((emb1+emb2)*0.5+Xcom)
        # output = self.MLP(emb)
        output1 = self.m1(emb)
        output = self.m2(output1)
        pslb = output

        cluster_assignment = torch.argmax(pslb, -1)
        # eq = torch.FloatTensor(torch.equal(cluster_assignment, torch.transpose(cluster_assignment,0,1)))
        onesl=torch.ones(pslb.shape[0]).cuda()
        zerosl=torch.zeros(pslb.shape[0]).cuda()
        weight_label=0
        # print(weight_label)
        cluster_assignment1= F.one_hot(cluster_assignment,self.nclass)
        label = torch.topk(cluster_assignment1, 1)[1].squeeze(1)
        print(label)
        # print('label',label)
        # output = self.MLP(emb)
        # print(output1.unsqueeze(1).shape,'output1.unsqueeze(1)')
        # print(self.cluster_layer.shape,'output1.unsqueeze(1)')
        s = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        s = s.pow((1 + 1.0) / 2.0)
        s = (s.t() / torch.sum(s, 1)).t()
        return output, s, emb1, com1, com1, emb2, emb,self.estimated_adj,output1,label,weight_label
