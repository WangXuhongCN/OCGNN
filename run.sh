#!/bin/bash
#SBATCH -J wxh
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --gres=gpu:1

source activate torch


#python main.py --dataset PROTEINS_full --module GAT --gpu 0 --n-epochs 10000 --early-stop
python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 46
python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 58
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 666
# python main.py --dataset cora --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 1122

# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 46
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 52
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 58
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 666
# python main.py --dataset citeseer --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 1122

# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 46
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 52
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 58
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 666
# python main.py --dataset pubmed --module GAE --gpu 0 --n-epochs 5000 --early-stop --seed 1122
