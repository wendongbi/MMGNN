
from builtins import NotImplementedError
from functools import reduce
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from typing import Union, Tuple
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_sparse
from utils import adj_norm


class MM_Conv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True,
                 moment=1, 
                 mode = 'mean', #  cat or sum
                 moment_att_dim = 16,
                 use_adj_norm = True,
                 N = None, # num of samples
                 device = None, 
                 use_center_moment = None,
                 use_norm = False,
                 **kwargs):  # yapf: disable
        if use_adj_norm:
            kwargs.setdefault('aggr', 'add')
        else:
            kwargs.setdefault('aggr', 'mean')
        super(MM_Conv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.moment = moment
        self.mode = mode
        self.use_norm = use_norm
        self.moment_att_dim = moment_att_dim
        self.use_adj_norm = use_adj_norm
        self.use_center_moment = use_center_moment
        assert use_center_moment is not None
        
        if use_norm:
            self.norm = nn.LayerNorm(out_channels, elementwise_affine=True) 
        print('mode:', mode)
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        if mode == 'mlp': 
            # scheme 2
            self.lin_moment_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))
            print(moment, len(self.lin_moment_list))
        elif mode == 'attention':
            # scheme 2
            self.lin_self_out = Linear(in_channels[0], out_channels, bias=bias)
            self.lin_self_query = Linear(out_channels, moment_att_dim, bias=False)
            self.lin_key = Linear(out_channels, moment_att_dim, bias=False)
            self.lin_moment_list = nn.ModuleList()
            for _ in range(moment):
                self.lin_moment_list.append(Linear(in_channels[0], out_channels, bias=bias))
            print(moment, len(self.lin_moment_list))
            self.w_att = nn.Parameter(torch.FloatTensor(2 * moment_att_dim, out_channels))
        else:
            raise NotImplementedError

        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_norm:
            self.norm.reset_parameters()
        if self.mode == 'mlp':
            for fc in self.lin_moment_list:
                fc.reset_parameters()
        elif self.mode == 'attention':
            self.lin_self_out.reset_parameters()
            self.lin_self_query.reset_parameters()
            nn.init.xavier_uniform_(self.w_att.data, gain=1.414)
            for fc in self.lin_moment_list:
                fc.reset_parameters()
            self.lin_key.reset_parameters()
        else:
            raise NotImplementedError
    
        if self.root_weight:
            self.lin_r.reset_parameters()

    def get_attention_layer(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_row_col:Adj, size: Size = None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out_list = self.propagate(edge_index, x=x, size=size, moment=self.moment)
        if self.mode in  ['attention']:
            h_list = []
            k_list = []
            h_self = self.lin_self_out(x[0])
            if self.use_norm:
                h_self = self.norm(h_self) # ln
            q = self.lin_self_query(h_self).repeat(self.moment + 1, 1) # N * (m+1), D
            k0 = self.lin_key(h_self)
            h_list.append(h_self)
            k_list.append(k0)
            # output for each moment of 1st-neighbors
            for idx, fc in enumerate(self.lin_moment_list):
                h = fc(out_list[idx])
                if self.use_norm:
                    h = self.norm(h) # ln
                h_list.append(h)
                k = self.lin_key(h_list[-1])
                k_list.append(k)
            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.matmul(attn_input, self.w_att)) # N*(m+1), 1
            attention = F.softmax(e.view(len(k_list), -1, self.out_channels).transpose(0, 1), dim=1) # N, m+1, D
        else:
            return None
        return attention

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_row_col:Adj, thres_deg=0,
                size: Size = None, get_moment=False) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out_list = self.propagate(edge_index, x=x, size=size, moment=self.moment)
        if self.mode == 'mlp':
            # scheme 2
            out = None
            for idx, fc in enumerate(self.lin_moment_list):
                if out == None:
                    out = fc(out_list[idx])
                else:
                    out += fc(out_list[idx])
            out /= self.moment

            if thres_deg > 0:  
                out_1st = fc(out_list[0])
                deg = degree(edge_row_col[1], x[0].shape[0], dtype=x[0].dtype)
                mask_deg = deg < thres_deg
                out[mask_deg, :] = out_1st[mask_deg, :]

        elif self.mode == 'attention':
            h_list = []
            k_list = []
            h_self = self.lin_self_out(x[0])
            if self.use_norm:
                h_self = self.norm(h_self) # ln
            q = self.lin_self_query(h_self).repeat(self.moment + 1, 1) # N * (m+1), D
            k0 = self.lin_key(h_self)
            h_list.append(h_self)
            k_list.append(k0)
            # output for each moment of 1st-neighbors
            for idx, fc in enumerate(self.lin_moment_list):
                h = fc(out_list[idx])
                if self.use_norm:
                    h = self.norm(h) # ln
                h_list.append(h)
                k = self.lin_key(h_list[-1])
                k_list.append(k)

            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.matmul(attn_input, self.w_att)) # N*(m+1), 1
            attention = F.softmax(e.view(len(k_list), -1, self.out_channels).transpose(0, 1), dim=1) # N, m+1, D
            out = torch.stack(h_list, dim=1).mul(attention).sum(1) # N, D
        else:
            raise NotImplementedError


        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)
            

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out, out_list
        # return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j
    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor,
                              moment: int) -> Tensor:
        mu = matmul(adj_t, x[0], reduce=self.aggr)
        out_list = [mu]
        if moment > 1:
            if self.use_center_moment:
                sigma = matmul(adj_t, (x[0] - mu).pow(2), reduce=self.aggr)
            else:
                sigma = matmul(adj_t, (x[0]).pow(2), reduce=self.aggr)
            
            sigma[sigma == 0] = 1e-16
            sigma = sigma.sqrt()
            out_list.append(sigma)


            for order in range(3, moment+1):
                gamma = matmul(adj_t, x[0].pow(order), reduce=self.aggr)
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

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
