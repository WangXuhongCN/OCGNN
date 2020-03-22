import time
import numpy as np
import torch
import logging
from dgl.contrib.sampling.sampler import NeighborSampler
# import torch.nn as nn
# import torch.nn.functional as F



from optim.loss import loss_function,init_center,get_radius

from utils.evaluate import evaluate

def train(args,data,model):

    checkpoints_path=f'./checkpoints/{args.dataset}+OC-{args.module}+bestcheckpoint.pt'

    logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    logger=logging.getLogger('OCGNN')
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

    input_feat=data['features']
    input_g=data['g']

    data_center= init_center(args,input_g,input_feat, model)
    radius=torch.tensor(0, device=f'cuda:{args.gpu}')# radius R initialized with 0 by default.

    #train_inputs=data['features']

    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        #model.train()
        if epoch %5 == 0:
            t0 = time.time()
        # forward

        outputs= model(input_g,input_feat)
        
        loss,dist,_=loss_function(args.nu, data_center,outputs,data['train_mask'],radius)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%5 == 0:
            dur.append(time.time() - t0)
            radius.data=torch.tensor(get_radius(dist, args.nu), device=f'cuda:{args.gpu}')



        auc,ap,f1,acc,precision,recall,the_loss = evaluate(args,model, data_center,data,radius,'val')
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val AUROC {:.4f} | Val F1 {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                            auc,f1, data['n_edges'] / np.mean(dur) / 1000))
        if args.early_stop:
            if stopper.step(auc,float(the_loss.cpu().numpy()), model,checkpoints_path):   
                break

    #model_path=checkpoints_path+f'{epoch}+bestcheckpoint.pt'
    print()
    if args.early_stop:
        print(f'model loaded.')
        model.load_state_dict(torch.load(checkpoints_path))

    auc,ap,f1,acc,precision,recall,_ = evaluate(args,model, data_center,data,radius,'test')
    print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auc,ap))
    print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
    logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    logger.info('\n')
    return model


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.lowest_loss = None
        self.early_stop = False

    def step(self, acc,loss, model,path):
        score = acc
        cur_loss=loss
        if (self.best_score is None) or (self.lowest_loss is None):
            self.best_score = score
            self.lowest_loss = cur_loss
            self.save_checkpoint(acc,loss,model,path)
        elif (score < self.best_score) and (cur_loss > self.lowest_loss):
            self.counter += 1
            if self.counter >= 0.8*(self.patience):
                print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.save_checkpoint(acc,loss,model,path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, acc,loss,model,path):
        '''Saves model when validation loss decrease.'''
        print(f'model saved. loss={loss} AUC={acc}')
        torch.save(model.state_dict(), path)