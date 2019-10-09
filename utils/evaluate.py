from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve
import torch
from optim.loss import loss_function
import numpy as np

def evaluate(args,model, data_center,features, labels, mask,radius):
    
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        #outputs = outputs[mask]
        labels = labels[mask]
        _ , _,scores=loss_function(args,data_center,outputs,mask,radius)
        
        labels=labels.cpu().numpy()
        scores=scores.cpu().numpy()
        # print(scores.min())
        # print(scores.max())
        # print(scores.mean())
        auroc=roc_auc_score(labels,scores)
        auprc=average_precision_score(labels,scores)
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        #return correct.item() * 1.0 / len(labels)
    
    #metric={}

    return auroc,auprc