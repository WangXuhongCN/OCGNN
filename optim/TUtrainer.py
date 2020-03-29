import time
import numpy as np
import torch
import os
import torch.nn as nn
import logging
#from dgl.contrib.sampling.sampler import NeighborSampler
# import torch.nn as nn
# import torch.nn.functional as F



from optim.loss import loss_function,init_center,get_radius,EarlyStopping

from utils.evaluate import multi_graph_evaluate

def train(args, logger,dataset, model, val_dataset=None,path=None):
    '''
    training function
    '''
    checkpoints_path=path

    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    logger.info('Start training')
    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')
    
    dataloader = dataset
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
    #                                     model.parameters()), lr=0.001)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    #early_stopping_logger = {"best_epoch": -1, "val_acc": -1}


    #data_center= init_center(args,input_g,input_feat, model)
    data_center= torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')
    radius=torch.tensor(0, device=f'cuda:{args.gpu}')# radius R initialized with 0 by default.
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.n_epochs):
        begin_time = time.time()
        # accum_correct = 0
        # total = 0
        print("EPOCH ###### {} ######".format(epoch))
        computation_time = 0.0
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
            if torch.cuda.is_available():
                for (key, value) in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.cuda()
                #graph_labels = graph_labels.cuda()
            #print(batch_graph)
            train_mask=~batch_graph.ndata['node_labels'].bool().squeeze()
            model.zero_grad()
            compute_start = time.time()

            normlizing = nn.BatchNorm1d(batch_graph.ndata['node_attr'].shape[1], affine=False).cuda()
            input_attr=normlizing(batch_graph.ndata['node_attr'])

            # normlizing = nn.InstanceNorm1d(batch_graph.ndata['node_attr'].shape[1], affine=False).cuda()
            # input_attr=normlizing(batch_graph.ndata['node_attr'].unsqueeze(1)).squeeze()

            #data_center= init_center(args,batch_graph,batch_graph.ndata['node_attr'], model)
            #print('data_center',data_center)
            outputs = model(batch_graph,input_attr)
            # print('outputs mean',outputs.mean())
            # print('outputs std',outputs.std())

            #loss = loss_fn(outputs, batch_graph.ndata['node_labels'].float())
            loss,dist,score=loss_function(args.nu, data_center,outputs,radius,train_mask)
            #if batch_idx<=3:
                #print(dist)
                # print(score)
            # print('dist mean',dist.mean())
            # print('dist std',dist.std())
            loss.backward()
            batch_compute_time = time.time() - compute_start
            computation_time += batch_compute_time
            #nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            #radius.data=torch.tensor(get_radius(dist, args.nu), device=f'cuda:{args.gpu}')
            print('RRR',radius.data)
            print("Epoch {:05d},loss {:.4f} with {}-th batch time(s) {:.4f}".format(
            epoch, loss.item(), batch_idx, computation_time))
        #train_accu = accum_correct / total
        #print("train loss for this epoch {} is {}%".format(epoch,train_accu * 100))
        elapsed_time = time.time() - begin_time
        #print("Epoch {:05d}, loss {:.4f} with epoch time(s) {:.4f}".format(epoch,loss.item(), elapsed_time))
        if val_dataset is not None:
            auc,ap,f1,acc,precision,recall,loss = multi_graph_evaluate(args,checkpoints_path, model, data_center,val_dataset,radius,'val')
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val AUROC {:.4f} | Val F1 {:.4f} | Val ACC {:.4f} | ". format(
                epoch, elapsed_time, loss.item()*100000, auc,f1,acc))
            torch.cuda.empty_cache()
            if args.early_stop:
                if stopper.step(auc,loss.item()*100000, model,epoch,checkpoints_path):  
                    print("best epoch is EPOCH {}, val_auc is {}%".format(stopper.best_epoch,
                                                        stopper.best_score)) 
                    break

    # auc,ap,f1,acc,precision,recall,_ = multi_graph_evaluate(args,checkpoints_path, model, data_center,data,radius,'test')
    # torch.cuda.empty_cache()
    # print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auc,ap))
    # print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    # logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
    # logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    # logger.info('\n')
    return model    
