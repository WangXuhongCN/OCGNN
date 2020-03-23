import argparse
from dgl.data import register_data_args
import logging
import fire
from optim import trainer, TUtrainer
from optim.loss import loss_function,init_center
from datasets import dataloader,TUloader
from networks.init import init_model
import numpy as np
import torch
from dgl import random as dr

def main(args):
    if args.seed!=-1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        #torch.backends.cudnn.deterministic=True
        dr.seed(args.seed)

    if args.dataset in 'TU_PROTEINS_full':
        train_loader, val_loader, test_loader, input_dim, label_dim=TUloader.loader(args)
        model=init_model(args,input_dim)
        model=TUtrainer.train(args,train_loader,model,val_dataset=None)
    else:  
        data=dataloader.loader(args)
        model=init_model(args,data['input_dim'])
        model=trainer.train(args,data,model)
# valuation(args,datadict,model)
# test(args,data,model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCGNN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--nu", type=float, default=0.1,
            help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--seed", type=int, default=52,
            help="random seed, -1 means dont fix seed")
    parser.add_argument("--module", type=str, default='GraphSAGE',
            help="GCN/GAT/GIN/GraphSAGE")
    parser.add_argument('--n-worker', type=int,default=1,
            help='number of workers when dataloading')
    parser.add_argument('--batch-size', type=int,default=32,
            help='number of workers when dataloading')
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=5000,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--norm", action='store_true',
            help="graph normalization (default=False)")
    parser.set_defaults(self_loop=False)
    parser.set_defaults(norm=False)
    args = parser.parse_args()
    if args.module=='GCN':
        args.self_loop=True
        args.norm=True
    if args.module=='GAT':
        args.self_loop=True
    if args.module=='GraphSAGE':
        args.self_loop=True
    fire.Fire(main(args))
