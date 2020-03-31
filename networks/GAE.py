import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from networks.GCN import GCN

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def forward(self, z, sigmoid=True):
        """Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class GAE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GAE, self).__init__()
        #self.g = g
        self.encoder=GCN(g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout)
        self.A_decoder=GCN(g,
                 n_classes,
                 n_hidden,
                 in_feats,
                 n_layers,
                 activation,
                 dropout)
        self.S_decoder=GCN(g,
                 n_classes,
                 n_hidden,
                 in_feats,
                 n_layers-1,
                 activation,
                 dropout)
        self.InnerProducter=InnerProductDecoder()

    def forward(self,g, features):
        h = features
        z=self.encoder(g, features)
        recon=self.A_decoder(g, z)
        z_=self.S_decoder(g,z) 
        adj=self.InnerProducter(z_)
        return z,recon,adj

# class GCN(nn.Module):
#     def __init__(self,
#                  g,
#                  in_feats,
#                  n_hidden,
#                  n_classes,
#                  n_layers,
#                  activation,
#                  dropout):
#         super(GCN, self).__init__()
#         self.g = g
#         self.layers = nn.ModuleList()
#         # input layer
#         self.layers.append(GraphConv(in_feats, n_hidden, bias=True, activation=activation))
#         # hidden layers
#         for i in range(n_layers - 1):
#             self.layers.append(GraphConv(n_hidden, n_hidden,  bias=True, activation=activation))
#         # output layer
#         self.layers.append(GraphConv(n_hidden, n_classes,bias=True))
#         self.dropout = nn.Dropout(p=dropout)
#         print(self.layers)
#     def forward(self,g, features):
#         h = features
#         for i, layer in enumerate(self.layers):
#             if i != 0:
#                 h = self.dropout(h)
#             h = layer(g, h)
#         return h