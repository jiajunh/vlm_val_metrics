#!/bin/bash

#SBATCH --job-name=llava_baseline_vqa
#SBATCH --mem=24G
#SBATCH -t 0-01:00
#SBATCH -p gpu_requeue
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/llava_vqa_baseline_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/llava_vqa_baseline_%j.err
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

mamba activate llava

python --version

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/llava

nvidia-smi

python llava.py
