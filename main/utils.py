import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
import os
from networkx.readwrite import json_graph
import pdb
from dataset import Facebook100
import random
# from model import HM_GCNNet, HM_SAGENet
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import os.path as osp
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Flickr
import numpy as np
import sys
import scipy
from torch.nn.utils import clip_grad_norm_
sys.setrecursionlimit(99999)

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



def pyg_dataset(args, transform=None):
    if args.dataset == 'Flickr':
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
    return dataset



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_dataset(args):
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        return load_dataset_citation(args.dataset)
    else:
        dataset = pyg_dataset(args)
        data = dataset[0]
        adj = scipy.sparse.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index), shape=(data.num_nodes, data.num_nodes))
        adj = sys_normalized_adjacency(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = data.x
        labels = data.y
        idx_train = torch.where(data.train_mask)[0]
        idx_val = torch.where(data.val_mask)[0]
        idx_test = torch.where(data.test_mask)[0]
        return adj, features, labels, idx_train, idx_val, idx_test
        
        
        
# adapted from tkipf/gcn
def load_dataset_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # print(type(adj))
    adj = sys_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # print(features.shape)
    return adj, features, labels, idx_train, idx_val, idx_test
        

# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        #for v in range(nb_nodes):
        for v in adj[u,:].nonzero()[1]:
            #if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)

def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret

def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
              #  if adj[i,j] == 1:
                 return False
    return True

def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits={}
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i]==0 or mapping[j]==0:
                dict_splits[0]=None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]]='test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label='test'
                        elif ds_label[i]['val']:
                            ind_label='val'
                        else:
                            ind_label='train'
                        if dict_splits[mapping[i]]!= ind_label:
                            print ('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print ('label of both nodes different, exiting!!')
                    return None
    return dict_splits