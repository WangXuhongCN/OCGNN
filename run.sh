#!/bin/bash
#SBATCH -J wxh
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --gres=gpu:1

source activate torch


python main.py --dataset PROTEINS_full --module GraphSAGE --gpu 0 --n-epochs 200 --early-stop
# python main.py --dataset citeseer --module GraphSAGE --gpu 0 --n-epochs 1 --early-stop
# python main.py --dataset pubmed --module GraphSAGE --gpu 0 --n-epochs 1 --early-stop