import argparse
from dgl.data import register_data_args
import logging
import fire
from optim.trainer import train
from optim.loss import loss_function,init_center
from datasets.dataloader import dataloader
from utils.evaluate import evaluate
from networks.init import init_model
import numpy as np
import torch
def main(args):
	SEED=46
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

	datadict=dataloader(args)
	model=init_model(args,datadict)
	model=train(args,datadict,model)
# valuation(args,datadict,model)
# test(args,data,model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--nu", type=float, default=0.1,
            help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--module", type=str, default='GCN',
            help="GCN/GraphSAGE")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
