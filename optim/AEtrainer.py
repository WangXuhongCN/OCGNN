import time
import numpy as np
import torch
import logging
#from dgl.contrib.sampling.sampler import NeighborSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve


from optim.loss import EarlyStopping


def train(args,logger,data,model,path):

    checkpoints_path=path

    # logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    # logger=logging.getLogger('OCGNN')
    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    logger.info('Start training')
    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    # initialize data center

    adj=data['g'].adjacency_matrix().to_dense().cuda()
    loss_fn = nn.MSELoss()
    #train_inputs=data['features']
    print('adj dim',adj[data['train_mask']].size())

    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        #model.train()
        if epoch %5 == 0:
            t0 = time.time()
        # forward

        z,re_x,re_adj= model(data['g'],data['features'])

        loss=Dom_loss(re_x,re_adj,adj,data['features'],data['train_mask'],loss_fn)
        #loss,dist,_=loss_fn(args.nu, data_center,outputs,radius,data['train_mask'])



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%5 == 0:
            dur.append(time.time() - t0)
        
        auc,ap,val_loss=fixed_graph_evaluate(args,model,data,adj,data['val_mask'])
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val AUROC {:.4f} | Val loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item()*100000,
                                            auc,val_loss, data['n_edges'] / np.mean(dur) / 1000))
        if args.early_stop:
            if stopper.step(auc,val_loss.item(), model,epoch,checkpoints_path):   
                break

    print(f'loading model before testing.')
    model.load_state_dict(torch.load(checkpoints_path))

    auc,ap,_ = fixed_graph_evaluate(args,model,data,adj,data['test_mask'])
    print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auc,ap))
    #print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    # logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
    # logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    # logger.info('\n')
    return model

def Dom_loss(re_x,re_adj,adj,x,mask,loss_fn):
    A_loss=loss_fn(re_x[mask], x[mask])
    S_loss=loss_fn(re_adj[mask], adj[mask])
    return 0.8*A_loss + 0.2*S_loss

def fixed_graph_evaluate(args,model,data,adj,mask):
    loss_fn = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        
        z,re_x,re_adj= model(data['g'],data['features'])

        labels = data['labels'][mask]
        
        loss_mask=mask.bool() & data['labels'].bool()
        #print(loss_mask.)
        loss=Dom_loss(re_x,re_adj, adj, data['features'],loss_mask,loss_fn)
        #print(recon[data['val_mask']].size())
        A_scores=F.mse_loss(re_x[mask], data['features'][mask], reduce=False)
        S_scores=F.mse_loss(re_adj[mask], adj[mask], reduce=False)
        scores=torch.mean(A_scores,1)+torch.mean(S_scores,1)

        labels=labels.cpu().numpy()
        # print(labels.shape)
        # print(scores.shape)
        #dist=dist.cpu().numpy()
        scores=scores.cpu().numpy()
        #pred=thresholding(scores,0)
        #print('scores.shape',scores)
        auc=roc_auc_score(labels, scores)
        ap=average_precision_score(labels, scores)

        # acc=accuracy_score(labels,pred)
        # recall=recall_score(labels,pred)
        # precision=precision_score(labels,pred)
        # f1=f1_score(labels,pred)


    return auc,ap,loss