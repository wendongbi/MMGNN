import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MMConv(nn.Module):
    def __init__(self, in_features, out_features,  moment=3, use_center_moment=False):
        super(MMConv, self).__init__() 
        self.moment = moment
        self.use_center_moment = use_center_moment
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.w_att = Parameter(torch.FloatTensor(self.in_features * 2,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.w_att.data.uniform_(-stdv, stdv)
    def moment_calculation(self, x, adj_t, moment):
        mu = torch.spmm(adj_t, x)
        out_list = [mu]
        if moment > 1:
            if self.use_center_moment:
                sigma = torch.spmm(adj_t, (x - mu).pow(2))
            else:
                sigma = torch.spmm(adj_t, (x).pow(2))
            sigma[sigma == 0] = 1e-16
            sigma = sigma.sqrt()
            out_list.append(sigma)

            for order in range(3, moment+1):
                gamma = torch.spmm(adj_t, x.pow(order))
                mask_neg = None
                if torch.any(gamma == 0):
                    gamma[gamma == 0] = 1e-16
                if torch.any(gamma < 0):
                    mask_neg = gamma < 0
                    gamma[mask_neg] *= -1
                gamma = gamma.pow(1/order)
                if mask_neg != None:
                    gamma[mask_neg] *= -1
                out_list.append(gamma)
        return out_list
    def attention_layer(self, moments, q):
            k_list = []
            # if self.use_norm:
            #     h_self = self.norm(h_self) # ln
            q = q.repeat(self.moment, 1) # N * m, D
            # output for each moment of 1st-neighbors
            k_list = moments
            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.mm(attn_input, self.w_att)) # N*m, D
            attention = F.softmax(e.view(len(k_list), -1, self.out_features).transpose(0, 1), dim=1) # N, m, D
            out = torch.stack(k_list, dim=1).mul(attention).sum(1) # N, D
            return out
    def forward(self, input, adj , h0 , lamda, alpha, l, beta=0.1):
        theta = math.log(lamda/l+1)
        h_agg = torch.spmm(adj, input)
        h_agg = (1-alpha)*h_agg+alpha*h0
        h_i = torch.mm(h_agg, self.weight)
        h_i = theta*h_i+(1-theta)*h_agg
        # h_moment = self.attention_layer(self.moment_calculation(input, adj, self.moment), h_i)
        h_moment = self.attention_layer(self.moment_calculation(h0, adj, self.moment), h_i)
        output = (1 - beta) * h_i + beta * h_moment
        return output