
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
import os.path as osp
import os
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Flickr
print(torch_geometric.__version__)
import numpy as np
import sys
from torch.nn.utils import clip_grad_norm_
sys.path.append('./')
from dataset import Facebook100
import random
# from model import HM_GCNNet, HM_SAGENet
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def adj_norm(adj, norm='row'):
    if not adj.has_value():
        adj = adj.fill_value(1., dtype=None)
    # add self loop
    adj = fill_diag(adj, 0.)
    adj = fill_diag(adj, 1.)
    deg = sparsesum(adj, dim=1)
    if norm == 'symmetric':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
        adj = mul(adj, deg_inv_sqrt.view(1, -1)) # col normalization
    elif norm == 'row':
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj = mul(adj, deg_inv_sqrt.view(-1, 1)) # row normalization
    else:
        raise NotImplementedError('Not implete adj norm: {}'.format(norm))
    return adj


def onehot_encoder_dim(values):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoding = onehot_encoder.fit_transform(integer_encoded)
    return torch.from_numpy(onehot_encoding)

def onehot_encoder(x):
    x_onehot = None
    for col_idx in range(x.shape[1]):
        col = onehot_encoder_dim(x[:, col_idx])
        if x_onehot is None:
            x_onehot = col
        else:
            x_onehot = torch.cat([x_onehot, col], dim=1)
    return x_onehot.float()



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def col_norm(x, mode='std'):
    assert isinstance(x, torch.Tensor)
    if mode == 'std':
        mu = x.mean(dim=0)
        sigma = x.std(dim=0)
        out = (x - mu) / sigma
    return out

    

def train(data, model, optimizer, clip_grad=False):
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask])
    loss.backward()
    if clip_grad:
        clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach().item()


# @torch.no_grad()
def test(data, model, dataset_name):
    with torch.no_grad():
        model.eval()
        log_probs, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = log_probs[mask].max(1)[1]
            score = pred.eq(data.y[mask]).sum().item() / mask.sum().item() # acc
            accs.append(score)
        return accs


def build_model(args, dataset, device: torch.device, model_init_seed=None):
    if model_init_seed is not None:
        set_random_seed(model_init_seed)
    if args.model == 'MM_GNN':
        from model import MM_GNN
        model = MM_GNN(dataset, args.num_layer, args.moment, use_norm=args.use_norm, hidden=args.hidden, mode=args.mode, use_adj_norm=args.use_adj_norm, use_center_moment=args.use_center_moment, device=device)
    else:
        raise NotImplementedError('Not implemented model: {}'.format(args.model))
    return model.to(device)

def build_dataset(args, transform=None):
    data_split = None
    if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        path = osp.join(args.data_dir, args.dataset)
        dataset = Planetoid(path, args.dataset, transform=transform, split='public')
    elif args.dataset == 'Flickr':
        path = osp.join(args.data_dir, args.dataset)
        dataset = Flickr(path, transform=transform)
    elif args.dataset in ['chameleon', 'squirrel']:
        from torch_geometric.datasets import WikipediaNetwork
        path = osp.join(args.data_dir, args.dataset)
        dataset = WikipediaNetwork(path, args.dataset)
    else:
        # for facebook 100 dataset
        path = osp.join(args.data_dir, args.dataset)
        dataset = Facebook100(path, args.dataset, transform=transform, split='random', \
            num_val=500, num_test=None, num_train_per_class=200)
    return dataset, data_split