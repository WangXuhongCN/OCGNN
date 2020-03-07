#!/bin/bash
#SBATCH -J wxh
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH --gres=gpu:1

source activate torch


python main.py --dataset cora --module GraphSAGE --gpu 0 --n-epochs 1000 --early-stop