import argparse
from dgl.data import register_data_args

from datasets.dataloader import baselinedataloader
from utils.evaluate import baseline_evaluate


from embedding.get_embedding import embedding
from pyod.models.ocsvm import OCSVM
import numpy as np
from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve

def main(args):
	datadict=baselinedataloader(args)

	if args.mode=='X':
		data=datadict['features']
		#print('X shape',data.shape)
	else:
		embeddings=embedding(args,datadict)
		if args.mode=='A':
			data=embeddings
			#print('A shape',data.shape)
		if args.mode=='AX':
			data=np.concatenate((embeddings,datadict['features']),axis=1)
			#print('AX shape',data.shape)


	clf = OCSVM(nu=args.nu,contamination=0.1)
	clf.fit(data[datadict['train_mask']])

	print('-------------Evaluating Validation Results--------------')
	y_pred_val=clf.predict(data[datadict['val_mask']])
	y_score_val=clf.decision_function(data[datadict['val_mask']])
	baseline_evaluate(datadict,y_pred_val,y_score_val,val=True)


	print('-------------Evaluating Test Results--------------')
	y_pred_test=clf.predict(data[datadict['test_mask']])
	y_score_test=clf.decision_function(data[datadict['test_mask']])
	baseline_evaluate(datadict,y_pred_test,y_score_test,val=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--mode", type=str, default='A',choices=['A','AX','X'],
            help="dropout probability")
    parser.add_argument("--normal-class", type=int, default=2,
            help="normal-class")
    parser.add_argument("--nu", type=float, default=0.1,
            help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--emb-method", type=str, default='DeepWalk',
            help="embedding methods: DeepWalk, Node2Vec, LINE, SDNE, Struc2Vec")  
    parser.add_argument("--ad-method", type=str, default='OCSVM',
            help="embedding methods: LOF,OCSVM,IF")            
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
            help="Weight for L2 loss")
#     parser.add_argument("--self-loop", action='store_true',
#             help="graph self-loop (default=False)")
#     parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
