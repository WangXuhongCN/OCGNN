import time
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F



from optim.loss import loss_function,init_center,get_radius

from utils.evaluate import evaluate


def train(args,data,model):

    checkpoints_path=f'./checkpoints/{args.dataset}+OC-{args.module}+bestcheckpoint.pt'
    #loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer AdamW
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=200)
    # initialize data center
    data_center= init_center(args,data, model)
    radius=torch.tensor(0, device=f'cuda:{args.gpu}')# radius R initialized with 0 by default.

    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        #model.train()
        if epoch %5 == 0:
            t0 = time.time()
        # forward
        if args.module== 'GIN':
            outputs= model(data['g'],data['features'])
        else:
            outputs= model(data['features'])
        
        loss,dist,_=loss_function(args,data_center,outputs,data['train_mask'],radius)
        #loss=torch.mean(loss)
        #loss = loss_fcn(outputs[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%5 == 0:
            dur.append(time.time() - t0)
            radius.data=torch.tensor(get_radius(dist, args.nu), device=f'cuda:{args.gpu}')



        auc,ap,f1,acc,precision,recall = evaluate(args,model, data_center,data,radius,'val')
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val AUROC {:.4f} | Val F1 {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                            auc,f1, data['n_edges'] / np.mean(dur) / 1000))
        if args.early_stop and epoch>int(0.5*args.n_epochs):
            if stopper.step(auc, model,checkpoints_path):   
                break

    #model_path=checkpoints_path+f'{epoch}+bestcheckpoint.pt'
    print()
    if args.early_stop:
        print(f'model loaded.')
        model.load_state_dict(torch.load(checkpoints_path))

    auc,ap,f1,acc,precision,recall = evaluate(args,model, data_center,data,radius,'test')
    print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auc,ap))
    print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    return model


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model,path):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc,model,path)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= 0.8*(self.patience):
                print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc,model,path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, acc,model,path):
        '''Saves model when validation loss decrease.'''
        print(f'model saved. AUC={acc}')
        torch.save(model.state_dict(), path)