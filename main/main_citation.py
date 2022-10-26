from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import sys
sys.path.append('../models/torch')
from model import MMGNN
import uuid


def train():
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def valid():
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()
    
def test():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()

def test_final():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_epoch', type=int, default=600, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--wd1', type=float, default=0.01)
    parser.add_argument('--wd2', type=float, default=5e-4)
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--data_dir', default='../dataset', help='dateset directory')
    parser.add_argument('--dataset', default='cora', help='dateset name')
    parser.add_argument('--gpu', type=int, default=0, help='device id')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
    parser.add_argument('--moment', type=int, default=3, help='max moment used in multi-moment model(MMGNN)')
    parser.add_argument('--mode', type=str, default='attention', help='mode to combine different moments feats, choose from mlp or attention')
    parser.add_argument('--auto_fixed_seed', action='store_true', help='fixed random seed of each run by run_id(0, 1, 2, ...)')
    parser.add_argument('--use_adj_norm', action='store_true', help='whether use adj normalization(row or symmetric norm).')
    parser.add_argument('--split_idx', type=int, default=0, help='split idx of multi-train/val/test mask dataset')
    parser.add_argument('--use_center_moment', action='store_true', help='whether to use center moment for MMGNN')
    parser.add_argument('--use_norm', action='store_true', help='whether to use layer norm for MMGNN')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data
    adj, features, labels,idx_train,idx_val,idx_test = load_dataset(args)
    cudaid = "cuda:"+str(args.gpu)
    device = torch.device(cudaid)
    features = features.to(device)
    adj = adj.to(device)
    print(adj)
    checkpt_file = './ckpt/'+uuid.uuid4().hex+'.pt'
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
        
    print(cudaid,checkpt_file)

    model = MMGNN(nfeat=features.shape[1],
                    nlayers=args.layer,
                    nhidden=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    lamda = args.lamda, 
                    alpha=args.alpha,
                    use_center_moment=args.use_center_moment,
                    moment=args.moment).to(device)
    optimizer = optim.Adam([
                            {'params':model.params1,'weight_decay':args.wd1},
                            {'params':model.params2,'weight_decay':args.wd2},
                            ],lr=args.lr)
    t_total = time.time()
    bad_counter = 0
    best = float('inf')
    best_epoch = 0
    acc = 0
    for epoch in range(args.num_epoch):
        loss_tra,acc_tra = train()
        loss_val,acc_val = valid()
        if(epoch+1)%1 == 0: 
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100),
                '| test',
                'acc:{:.2f}'.format(100 * test()[1]))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    if args.test:
        acc = test_final()[1]
    print("Training time: {:.4f}s".format(time.time() - t_total))
    print('Best epoch id: {}'.format(best_epoch))
    print("Test" if args.test else "Val","acc.:{:.2f}".format(acc*100))
    





