# OCGNN
The anonymous implement of the paper (IJCAI paper id 1321). Since my code is a little "academic", I am still working on reconstructing my code. 

# Requirements
pytorch>=1.0  
DGL==0.4.1
sklearn>=0.20.1
numpy>=1.16
networkx>=2.1

# Example 
python main.py --dataset [cora/citeseer/pubmed] --module [GCN/GAT/GraphSAGE] --gpu 0 --n-epochs 1000 --early-stop
