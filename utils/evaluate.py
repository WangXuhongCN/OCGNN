from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve
import torch
from optim.loss import loss_function
import numpy as np

def evaluate(args,model, data_center,data,radius,mode='val'):
    
    model.eval()
    with torch.no_grad():


        outputs= model(data['g'],data['features'])

        
        if mode=='val':
            labels = data['labels'][data['val_mask']]
            _ , dist ,scores=loss_function(args,data_center,outputs,data['val_mask'],radius)
        if mode=='test':
            labels = data['labels'][data['test_mask']]
            _ , dist ,scores=loss_function(args,data_center,outputs,data['test_mask'],radius)

        labels=labels.cpu().numpy()
        dist=dist.cpu().numpy()
        scores=scores.cpu().numpy()

        pred=thresholding(scores,0)
        # print(scores.min())
        # print(scores.max())
        # print(scores.mean())
        auc=roc_auc_score(labels, scores)
        ap=average_precision_score(labels, scores)

        acc=accuracy_score(labels,pred)
        recall=recall_score(labels,pred)
        precision=precision_score(labels,pred)
        f1=f1_score(labels,pred)

    
    #metric={}

    return auc,ap,f1,acc,precision,recall

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

