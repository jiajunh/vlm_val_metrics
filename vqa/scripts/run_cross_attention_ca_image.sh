#!/bin/bash

#SBATCH --job-name=llava_cross_attention_ca_image
#SBATCH --mem=24G
#SBATCH -t 0-16:00
#SBATCH -p gpu_requeue
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/llava_cross_attention_ca_image_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/logs/llava_cross_attention_ca_image_%j.err
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

mamba activate llava

python --version

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/test_verl/llava

nvidia-smi

python llava.py \
    --image_dir './vqa_samples_1k_2/images/' \
    --annotation_file './vqa_samples_1k_2/annotations.json' \
    --question_file './vqa_samples_1k_2/questions.json' \
    --output_dir './vqa_results_2' \
    --use_cross_attention \
    --cross_attention_image \
    --layer_type "3c" \
    --target_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
