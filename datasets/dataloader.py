from dgl.data import load_data, tu
from dgl import DGLGraph, transform
import torch
import torch.utils.data
import numpy as np
import torch
import dgl
import networkx as nx
from datasets.prepocessing import one_class_processing

def loader(args):
    # load and preprocess dataset
    
    data = load_data(args)

    print(f'normal_class is {args.normal_class}')

    labels,train_mask,val_mask,test_mask=one_class_processing(data,args.normal_class,args)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    g = data.graph
        
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        #g=transform.remove_self_loop(g)
        #if args.module!='GraphSAGE':
        g.add_edges_from(zip(g.nodes(), g.nodes()))

    g = DGLGraph(g)
    n_edges = g.number_of_edges()
    if args.norm:
        
        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        if cuda:
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)

    datadict={'g':g,'features':features,'labels':labels,'train_mask':train_mask,
        'val_mask':val_mask,'test_mask': test_mask,'input_dim':in_feats,'n_classes':n_classes,'n_edges':n_edges}

    return datadict

def emb_dataloader(args):
    # load and preprocess dataset
    data = load_data(args)
    normal_class=args.normal_class
    labels,train_mask,val_mask,test_mask=one_class_processing(data,normal_class,args)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    g = data.graph


    datadict={'g':g,'features':features,'labels':labels,'train_mask':train_mask,
        'val_mask':val_mask,'test_mask': test_mask,'in_feats':in_feats,'n_classes':n_classes,'n_edges':n_edges}

    return datadict