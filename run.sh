#!/bin/bash
#SBATCH -J wxh
#SBATCH -p gpu

#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --gres=gpu:1

source activate torch


#python main.py --dataset PROTEINS_full --module GAT --gpu 0 --n-epochs 10000 

# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 46
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 52
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 58
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 666
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 1122

# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 46
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 52
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 58
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 666
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 1122

#python main.py --dataset pubmed --module GraphSAGE --gpu 0 --n-epochs 50  --seed 46
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000  --seed 52  --early-stop
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000  --seed 58 --early-stop
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000  --seed 666 --early-stop
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000  --seed 1122 --early-stop
python baseline.py --dataset pubmed --mode A --ad-method OCSVM
# python main.py --dataset cora --module GraphSAGE --gpu 0 --n-epochs 5000  --seed 46 --early-stop
# python main.py --dataset citeseer --module GraphSAGE --gpu 0 --n-epochs 5000  --seed 46 --early-stop
# python main.py --dataset pubmed --module GraphSAGE --gpu 0 --n-epochs 5000  --seed 46 --early-stop
#python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 52  --early-stop
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 58 --early-stop
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 666 --early-stop
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000  --seed 1122 --early-stop


# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 46 
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 52  
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 58 
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 666 
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000  --seed 1122 

# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 3000  --seed 52 
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 3000  --seed 58 
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 3000  --seed 666 
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 3000  --seed 1122 
