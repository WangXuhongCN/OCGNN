import networkx as nx
import os	
import numpy as np
from ge import DeepWalk	, Node2Vec, LINE, SDNE, Struc2Vec
from tqdm import tqdm
        
        
        
def embedding(args,datadict):  
    if not os.path.exists(f'./embedding/{args.dataset}.edgelist'):
        nx.write_edgelist(datadict['g'], f'./embedding/{args.dataset}.edgelist',data=[('weight',int)])
    datadict['g'] = nx.read_edgelist(f'./embedding/{args.dataset}.edgelist',create_using=nx.DiGraph(),nodetype=None,data=[('weight',int)])
    
    if not os.path.exists(f'./embedding/{args.dataset}_{args.emb_method}.emb'):
        if args.emb_method=='DeepWalk':
            model = DeepWalk(datadict['g'], walk_length=5, num_walks=50, workers=4)
            model.train(window_size=10, iter=10)
        if args.emb_method=='Node2Vec':
            model = Node2Vec(datadict['g'], walk_length = 10, num_walks = 80,p = 0.25, q = 4, workers = 4)
            model.train(window_size = 5, iter = 3)
        if args.emb_method=='LINE':
            model = LINE(datadict['g'],embedding_size=128,order='second')
            model.train(batch_size=1024,epochs=100,verbose=2)
        if args.emb_method=='SDNE':
            model = SDNE(datadict['g'],hidden_size=[256,128])
            model.train(batch_size=1024,epochs=100,verbose=2)
        if args.emb_method=='Struc2Vec':
            model = Struc2Vec(datadict['g'], walk_length=10, num_walks=80, workers=4, verbose=40, )
            model.train(window_size = 5, iter = 3)

        
        
        dict_embeddings = model.get_embeddings()
        embeddings=np.zeros((datadict['labels'].shape[0],dict_embeddings['0'].shape[0]))
        print('Saving the embeddings......')
        for key in tqdm(dict_embeddings):
            embeddings[int(key)] = dict_embeddings[key]
        np.savetxt(f'./embedding/{args.dataset}_{args.emb_method}.emb',embeddings)
        print(f'{embeddings.shape[1]}-dims Embeddings saved.')
    else:
        print('Loading the embeddings')
        embeddings=np.loadtxt(f'./embedding/{args.dataset}_{args.emb_method}.emb')
        print(f'{embeddings.shape[1]}-dims Embeddings load.')

    return embeddings