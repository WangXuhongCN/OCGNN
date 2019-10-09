import time
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F



from optim.loss import loss_function,init_center,get_radius

from utils.evaluate import evaluate


def train(args,data,model):


    #loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize data center
    data_center= init_center(args,data['features'], model,data['train_mask'])
    radius=torch.tensor(0, device=f'cuda:{args.gpu}')# radius R initialized with 0 by default.

    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        #model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        outputs= model(data['features'])
        
        loss,dist,_=loss_function(args,data_center,outputs,data['train_mask'],radius)
        #loss=torch.mean(loss)
        #loss = loss_fcn(outputs[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
            radius.data=torch.tensor(get_radius(dist, args.nu), device=f'cuda:{args.gpu}')



        auroc,auprc = evaluate(args,model, data_center,data['features'], data['labels'], data['val_mask'],radius)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                            auroc,auprc, data['n_edges'] / np.mean(dur) / 1000))

    print()
    auroc,auprc = evaluate(args,model, data_center,data['features'], data['labels'], data['test_mask'],radius)
    print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auroc,auprc))

    return model


