import numpy as np

def one_class_processing(labels,normal_class:int):
    labels,normal_idx,abnormal_idx=one_class_labeling(labels,normal_class)
    
    return one_class_masking(labels,normal_idx,abnormal_idx)


def one_class_labeling(labels,normal_class:int):
    normal_class=normal_class
    normal_idx=np.where(labels==normal_class)[0]
    abnormal_idx=np.where(labels!=normal_class)[0]
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)
    labels[normal_idx]=0
    labels[abnormal_idx]=1

    return labels,normal_idx,abnormal_idx

def one_class_masking(labels,normal_idx,abnormal_idx):
    train_mask=np.zeros(labels.shape)
    val_mask=np.zeros(labels.shape)
    test_mask=np.zeros(labels.shape)

    train_mask[normal_idx[:int(0.6*normal_idx.shape[0])]]=1

    val_mask[normal_idx[int(0.6*normal_idx.shape[0]):int(0.8*normal_idx.shape[0])]]=1
    val_mask[abnormal_idx[:int(0.5*abnormal_idx.shape[0])]]=1

    test_mask[normal_idx[int(0.8*normal_idx.shape[0]):]]=1
    test_mask[abnormal_idx[int(0.5*abnormal_idx.shape[0]):]]=1

    return labels,train_mask,val_mask,test_mask  