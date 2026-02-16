#!/bin/bash
#SBATCH -J 2dgaussian_test
#SBATCH --account=gts-yxie77-paid
#SBATCH --gres=gpu:1 -C H200
#SBATCH --mem=16GB
#SBATCH -t20
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

export TORCH_HOME=/storage/home/hcoda1/0/jlee3541/scratch/torch_cache
export XDG_CACHE_HOME=/storage/home/hcoda1/0/jlee3541/scratch/torch_cache

RUNPATH=$HOME/p-yxie77-0/projects/mfg_flow_matching/
cd $RUNPATH
module load anaconda3
conda activate mfg_flow
python mfg_flow_hd64.py
