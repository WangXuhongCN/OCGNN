import argparse
from dgl.data import register_data_args
import time
from datasets.dataloader import emb_dataloader
from utils.evaluate import baseline_evaluate
import fire
import logging
from embedding.get_embedding import embedding
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.auto_encoder import AutoEncoder


import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve

def main():
	parser = argparse.ArgumentParser(description='baseline')
	register_data_args(parser)
	parser.add_argument("--mode", type=str, default='A',choices=['A','AX','X'],
			help="dropout probability")
	parser.add_argument("--seed", type=int, default=-1,
            help="random seed, -1 means dont fix seed")
	parser.add_argument("--emb-method", type=str, default='DeepWalk',
			help="embedding methods: DeepWalk, Node2Vec, LINE, SDNE, Struc2Vec")  
	parser.add_argument("--ad-method", type=str, default='OCSVM',
			help="embedding methods: PCA,OCSVM,IF,AE")            
	args = parser.parse_args()
	
	if args.seed!=-1:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)

	logging.basicConfig(filename="./log/baseline.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
	logger=logging.getLogger('baseline')


	datadict=emb_dataloader(args)

	if args.mode=='X':
		data=datadict['features']
		#print('X shape',data.shape)
	else:
		t0 = time.time()
		embeddings=embedding(args,datadict)
		dur1=time.time() - t0
		
		if args.mode=='A':
			data=embeddings
			#print('A shape',data.shape)
		if args.mode=='AX':
			data=np.concatenate((embeddings,datadict['features']),axis=1)
			#print('AX shape',data.shape)

	logger.debug(f'data shape: {data.shape}')

	if args.ad_method=='OCSVM':
		clf = OCSVM(contamination=0.1)
	if args.ad_method=='IF':
		clf = IForest(n_estimators=100,contamination=0.1,n_jobs=-1,behaviour="new")
	if args.ad_method=='PCA':
		clf = PCA(contamination=0.1)
	if args.ad_method=='AE':
		clf = AutoEncoder(contamination=0.1)

	t1 = time.time()
	clf.fit(data[datadict['train_mask']])
	dur2=time.time() - t1

	print('traininig time:', dur1+dur2)

	logger.info('\n')
	logger.info('\n')
	logger.info(f'Parameters dataset:{args.dataset} datamode:{args.mode} ad-method:{args.ad_method} emb-method:{args.emb_method}')
	logger.info('-------------Evaluating Validation Results--------------')

	t2 = time.time()
	y_pred_val=clf.predict(data[datadict['val_mask']])
	y_score_val=clf.decision_function(data[datadict['val_mask']])
	auc,ap,f1,acc,precision,recall=baseline_evaluate(datadict,y_pred_val,y_score_val,val=True)
	dur3=time.time() - t2
	print('infer time:', dur3)

	logger.info(f'AUC:{round(auc,4)},AP:{round(ap,4)}')
	logger.info(f'f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')

	logger.info('-------------Evaluating Test Results--------------')
	y_pred_test=clf.predict(data[datadict['test_mask']])
	y_score_test=clf.decision_function(data[datadict['test_mask']])
	auc,ap,f1,acc,precision,recall=baseline_evaluate(datadict,y_pred_test,y_score_test,val=False)
	logger.info(f'AUC:{round(auc,4)},AP:{round(ap,4)}')
	logger.info(f'f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')


if __name__ == '__main__':

    #print(args)
	#main()
    fire.Fire(main)
