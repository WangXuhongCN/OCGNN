from dgl.data import load_data, tu
from dgl import DGLGraph, transform
import torch
import torch.utils.data
import numpy as np
import torch
import dgl
import networkx as nx
from datasets.prepocessing import one_class_processing, get_normal_class


def loader(args):
    #if args.dataset == 'PROTEINS_full':
    dataset = tu.TUDataset(name=args.dataset)
    train_size = int(args.train_ratio * len(dataset))
    test_size = int(args.test_ratio * len(dataset))
    val_size = int(len(dataset) - train_size - test_size)

    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset, (train_size, val_size, test_size))
    train_loader = prepare_dataloader(dataset_train, args, train=True)
    val_loader = prepare_dataloader(dataset_val, args, train=False)
    test_loader = prepare_dataloader(dataset_test, args, train=False)

    input_dim, label_dim, max_num_node = dataset.statistics()
    print("++++++++++STATISTICS ABOUT THE DATASET")
    print("dataset feature dimension is", input_dim)
    print("dataset label dimension is", label_dim)
    print("the max num node is", max_num_node)
    print("number of graphs is", len(dataset))


    return train_loader, val_loader, test_loader, input_dim, label_dim


def prepare_dataloader(dataset, args, train=False, pre_process=None):
    '''
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    '''
    if train:
        shuffle = True
    else:
        shuffle = False

    if pre_process:
        pre_process(dataset, args)

    # dataset.set_fold(fold)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=args.batch_size,
                                       shuffle=shuffle,
                                       collate_fn=batching_graph,
                                       drop_last=True,
                                       num_workers=args.n_worker)

def batching_graph(batch):
    '''
    for dataset batching
    transform ndata to tensor (in gpu is available)
    '''
    graphs, labels = map(list, zip(*batch))
    #cuda = torch.cuda.is_available()

    # batch graphs and cast to PyTorch tensor
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = torch.FloatTensor(value)
    batched_graphs = dgl.batch(graphs)

    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels))

    return batched_graphs, batched_labels