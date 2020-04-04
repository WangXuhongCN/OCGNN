# ECML-PKDD-2020-my-paper
The anonymous Pytorch and [DGL](https://github.com/dmlc/dgl) implement of the paper. 

## Details of our dataset
The Cora dataset has 7 categories of machine learning papers: "Case Based", "Genetic Algorithms", "**Neural Networks (Class label = 2 in the DGL dataloader)**", "Probabilistic Methods", "Reinforcement Learning", "Rule Learning", "Theory"; 

The Citeseer dataset consists of 6 paper classes: "Agents", "AI", "DB", "**IR (Class label = 3)**", "M"L, "HCI"; 

Each publication in the Pubmed dataset is classified into one of three classes ("Diabetes Mellitus, Experimental", "Diabetes Mellitus Type 1", "**Diabetes Mellitus Type 2 (Class label = 2)**"). 

In our experiments, classes in **bold** are defined as the normal classes, while the other classes are anomalous classes.

## GNN based methods
### Example:
python main.py --dataset [cora/citeseer/pubmed] --module [GCN/GAT/GraphSAGE/GAE] --nu 0.1 --lr 0.001 --n-hidden 32 --n-layers 2 --weight-decay 0.0005 --n-epochs 4000 --early-stop
### Requirements:
pytorch>=1.4
DGL>=0.4.2
sklearn>=0.20.1
numpy>=1.16
networkx>=2.1

## Two-stage mixture methods
### Example:
python twostage.py --dataset [cora / citeseer / pubmed] --mode [A/X/AX] --emb-method [DeepWalk / Node2Vec / LINE / SDNE / Struc2Vec] --ad-method [PCA / OCSVM / IF / AE]

### Requirements:
Pyod>=0.7.6
tensorflow>=1.4.0,<=1.12.0
gensim==3.6.0
DGL>=0.4.2
sklearn>=0.20.1
numpy>=1.16
networkx>=2.1