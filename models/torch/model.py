import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from convs import MMConv


class MMGNN(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, use_center_moment=False, moment=3):
        super(MMGNN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(MMConv(nhidden, nhidden, use_center_moment=use_center_moment, moment=moment))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
    
    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        _layers.append(h)
        for ind, conv in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act_fn(conv(h,adj,_layers[0],self.lamda,self.alpha, ind+1))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.fcs[-1](h)
        return F.log_softmax(h, dim=1)

    def get_emb(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        _layers.append(h)
        for ind, conv in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act_fn(conv(h,adj,_layers[0],self.lamda,self.alpha, ind+1))
        return h

if __name__ == '__main__':
    pass






