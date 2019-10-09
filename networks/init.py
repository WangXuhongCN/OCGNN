import torch.nn.functional as F
from .models import GCN

def init_model(args,datadict):
    # create GCN model
    model = GCN(datadict['g'],
                datadict['in_feats'],
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout)
                
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    if cuda:
        model.cuda()

    return model
    