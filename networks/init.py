import torch.nn.functional as F
from networks.GCN import GCN
from networks.GAT import GAT
from networks.GIN import GIN
from networks.GraphSAGE import GraphSAGE

def init_model(args,datadict):
    # create GCN model
    if args.module== 'GCN':
        model = GCN(datadict['g'],
                datadict['in_feats'],
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout)
    if args.module== 'GraphSAGE':
        model = GraphSAGE(datadict['g'],
                datadict['in_feats'],
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout,
                aggregator_type='pool')
    if args.module== 'GAT':
        model = GAT(datadict['g'],
                args.n_layers,
                datadict['in_feats'],
                args.n_hidden*2,
                args.n_hidden,
                heads=([8] * args.n_layers) + [1],
                activation=F.relu,
                feat_drop=args.dropout,
                attn_drop=args.dropout,
                negative_slope=0.2,
                residual=False)
    if args.module== 'GIN':
        model = GIN(num_layers=args.n_layers, 
                    num_mlp_layers=2, #1 means linear model.
                    input_dim=datadict['in_feats'], 
                    hidden_dim=args.n_hidden*2,
                    output_dim=args.n_hidden, 
                    final_dropout=args.dropout, 
                    learn_eps=True, 
                    graph_pooling_type="sum",
                    neighbor_pooling_type="sum")


    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    if cuda:
        model.cuda()

    return model
    