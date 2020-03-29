from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve
import torch
from optim.loss import loss_function,anomaly_score
import numpy as np
import torch.nn as nn

def fixed_graph_evaluate(args,path,model, data_center,data,radius,mode='val'):
    if mode=='test':
        print(f'model loaded.')
        model.load_state_dict(torch.load(path))
        
    model.eval()
    with torch.no_grad():
        outputs= model(data['g'],data['features'])  

        if mode=='val':
            labels = data['labels'][data['val_mask']]
            loss_mask=data['val_mask'].bool() & data['labels'].bool()
            #print(loss_mask.)
            _,scores=anomaly_score(data_center,outputs,radius,data['val_mask'])
            loss,_,_=loss_function(args.nu,data_center,outputs,radius,loss_mask)

        if mode=='test':
            labels = data['labels'][data['test_mask']]
            _,_,scores=loss_function(args.nu,data_center,outputs,radius,data['test_mask'])
        labels=labels.cpu().numpy()
        #dist=dist.cpu().numpy()
        scores=scores.cpu().numpy()
        pred=thresholding(scores,0)

        auc=roc_auc_score(labels, scores)
        ap=average_precision_score(labels, scores)

        acc=accuracy_score(labels,pred)
        recall=recall_score(labels,pred)
        precision=precision_score(labels,pred)
        f1=f1_score(labels,pred)

    if mode=='val':
        return auc,ap,f1,acc,precision,recall,loss
    if mode=='test':
        return auc,ap,f1,acc,precision,recall

def multi_graph_evaluate(args,path, model, data_center,dataloader,radius,mode='val'):
    '''
    evaluate function
    '''
    if mode=='test':
        print(f'model loaded.')
        model.load_state_dict(torch.load(path))
    model.eval()
    total_loss=0
    # pred_list=[]
    # labels_list=[]
    # scores_list=[]
    #correct_label = 0
    with torch.no_grad():
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            if torch.cuda.is_available():
                for (key, value) in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.cuda()
                #graph_labels = graph_labels.cuda()

            # normlizing = nn.InstanceNorm1d(batch_graph.ndata['node_attr'].shape[1], affine=False).cuda()
            # input_attr=normlizing(batch_graph.ndata['node_attr'].unsqueeze(1)).squeeze()

            normlizing = nn.BatchNorm1d(batch_graph.ndata['node_attr'].shape[1], affine=False).cuda()
            input_attr=normlizing(batch_graph.ndata['node_attr'])

            outputs = model(batch_graph,input_attr)

            labels = batch_graph.ndata['node_labels']
            #print(labels.size())
            loss_mask=~labels.bool().squeeze()
            #print(loss_mask.size())
            _,scores=anomaly_score(data_center,outputs,radius,mask=None)
            #print(outputs[loss_mask].size())
            loss,_,_=loss_function(args.nu,data_center,outputs,radius,loss_mask)

            # loss,_,scores=loss_function(args.nu,data_center,outputs,radius,mask=None)
            labels=labels.cpu().numpy().astype('int8')
            #dist=dist.cpu().numpy()
            scores=scores.cpu().numpy()
            pred=thresholding(scores,0)
            #print('pred',pred[:30])
            # print(labels[:10])
            # print(scores[:10])

            total_loss+=loss
            if batch_idx==0:
                labels_vec=labels
                pred_vec=pred
                scores_vec=scores
            else:
                pred_vec=np.append(pred_vec,pred)
                labels_vec=np.concatenate((labels_vec,labels),axis=0)
                scores_vec=np.concatenate((scores_vec,scores),axis=0)

        total_loss/=(batch_idx+1)
        print('score std',scores_vec.std())
        print('score mean',scores_vec.mean())
        print('labels mean',labels_vec.mean())
        print('pred mean',pred_vec.mean())
        auc=roc_auc_score(labels_vec, scores_vec)
        ap=average_precision_score(labels_vec, scores_vec)

        acc=accuracy_score(labels_vec,pred_vec)
        recall=recall_score(labels_vec,pred_vec)
        precision=precision_score(labels_vec,pred_vec)
        f1=f1_score(labels_vec,pred_vec)

    return auc,ap,f1,acc,precision,recall,total_loss


def thresholding(recon_error,threshold):
    ano_pred=np.zeros(recon_error.shape[0])
    for i in range(recon_error.shape[0]):
        if recon_error[i]>threshold:
            ano_pred[i]=1
    return ano_pred

def baseline_evaluate(datadict,y_pred,y_score,val=True):
    
    if val==True:
        mask=datadict['val_mask']
    if val==False:
        mask=datadict['test_mask']

    auc=roc_auc_score(datadict['labels'][mask],y_score)
    ap=average_precision_score(datadict['labels'][mask],y_score)
    acc=accuracy_score(datadict['labels'][mask],y_pred)
    recall=recall_score(datadict['labels'][mask],y_pred)
    precision=precision_score(datadict['labels'][mask],y_pred)
    f1=f1_score(datadict['labels'][mask],y_pred)

    return auc,ap,f1,acc,precision,recall

