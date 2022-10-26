import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_sparse
from convs import MM_Conv
from utils import adj_norm


class MMGNN(torch.nn.Module):
    def __init__(self, dataset, layer_num=2, moment=1, hidden=16, mode='sum', use_norm=False, use_adj_norm=False, use_adj_cache=True, device=None, use_center_moment=None):
        super(MMGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.adj_t_cache = None
        self.use_adj_cache = use_adj_cache
        self.use_adj_norm = use_adj_norm
        print('moment:', moment)
        # for ablation study
        self.out_layer_mode = mode
        self.out_layer_moment = moment

        if layer_num == 1:
            self.convs.append(
                MM_Conv(dataset.num_features, dataset.num_classes, use_norm=use_norm, moment=moment, mode=mode,  N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, device=device, use_center_moment=use_center_moment)
            )
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(MM_Conv(dataset.num_features, hidden, use_norm=use_norm, moment=moment, mode=mode,  N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, device=device, use_center_moment=use_center_moment))
                elif num == layer_num - 1:
                    self.convs.append(MM_Conv(hidden, dataset.num_classes, use_norm=use_norm, moment=self.out_layer_moment, mode=self.out_layer_mode,  N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm,  device=device, use_center_moment=use_center_moment))
                else:
                    self.convs.append(MM_Conv(hidden, hidden, use_norm=use_norm, moment=moment, mode=mode,  N=dataset[0].x.shape[0], use_adj_norm=self.use_adj_norm, device=device, use_center_moment=use_center_moment))
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def get_attention(self, data):
        x, edge_index = data.x, data.edge_index
        attention_bucket = []
        if isinstance(edge_index, torch.Tensor) and (self.adj_t_cache == None or not self.use_adj_cache):
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            if self.use_adj_norm:
                self.adj_t_cache = adj_norm(self.adj_t_cache, norm='row')

        for ind, conv in enumerate(self.convs):
            assert isinstance(conv, MM_Conv)
            if ind == len(self.convs) -1:
                attention_bucket.append(conv.get_attention_layer(x, self.adj_t_cache, edge_row_col=edge_index))
                x, moment_list = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)

            else:
                attention_bucket.append(conv.get_attention_layer(x, self.adj_t_cache, edge_row_col=edge_index))
                x, moment_list = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)
                x = F.dropout(F.relu(x), p=0.5, training=self.training)
        return attention_bucket
    def forward(self, data, get_moment=False):
        x, edge_index = data.x, data.edge_index
        if isinstance(edge_index, torch.Tensor) and (self.adj_t_cache == None or not self.use_adj_cache):
            self.adj_t_cache = torch_sparse.SparseTensor(row=edge_index[1], col=edge_index[0], value=torch.ones(data.edge_index.shape[1]).to(edge_index.device), sparse_sizes=(x.shape[0], x.shape[0]))
            if self.use_adj_norm:
                self.adj_t_cache = adj_norm(self.adj_t_cache, norm='row')
        if get_moment:
            moment_list_bucket = []
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) -1:
                x, moment_list = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)

                if get_moment:
                    moment_list_bucket.append(moment_list)
            else:
                x, moment_list = conv(x, self.adj_t_cache, edge_row_col=edge_index, thres_deg=0)

                x = F.dropout(F.relu(x), p=0.5, training=self.training)
                if get_moment:
                    moment_list_bucket.append(moment_list)

        if get_moment:
            return F.log_softmax(x, dim=1), moment_list_bucket
        return F.log_softmax(x, dim=1)
    
