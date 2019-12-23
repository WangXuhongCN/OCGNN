import networkx as nx
import os	
import numpy as np
from ge import DeepWalk	
from tqdm import tqdm
        
        
        
def embedding(args,datadict):  
    if not os.path.exists(f'./embedding/{args.dataset}.edgelist'):
        nx.write_edgelist(datadict['g'], f'./embedding/{args.dataset}.edgelist',data=[('weight',int)])

    datadict['g'] = nx.read_edgelist(f'./embedding/{args.dataset}.edgelist',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])

    if not os.path.exists(f'./embedding/{args.dataset}_{args.emb_method}.emb'):
        model = DeepWalk(datadict['g'], walk_length=10, num_walks=80, workers=8)
        model.train(window_size=5, iter=3)
        dict_embeddings = model.get_embeddings()
        embeddings=np.zeros((datadict['labels'].shape[0],dict_embeddings['0'].shape[0]))
        print('Saving the embeddings......')
        for key in tqdm(dict_embeddings):
            embeddings[int(key)] = dict_embeddings[key]
        np.savetxt(f'./embedding/{args.dataset}_{args.emb_method}.emb',embeddings)
        print('Embeddings saved.')
    else:
        print('Loading the embeddings')
        embeddings=np.loadtxt(f'./embedding/{args.dataset}_{args.emb_method}.emb')

    return embeddings