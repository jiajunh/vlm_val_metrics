#!/bin/bash

#SBATCH --job-name=llava_image_only_vqa
#SBATCH --mem=24G
#SBATCH -t 0-16:00
#SBATCH -p gpu_requeue
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/vlm_val/logs/llava_image_only_vqa_%j.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/vlm_val/logs/llava_image_only_vqa_%j.err
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

mamba activate llava

python --version

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/vlm_val/vqa/

nvidia-smi

# image only
python llava_mask.py \
    --image_dir './vqa_samples_1k_2/images/' \
    --annotation_file './vqa_samples_1k_2/annotations.json' \
    --question_file './vqa_samples_1k_2/questions.json' \
    --output_dir './vqa_results_2_image_only' \
    --use_cross_attention \
    --target_layers 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31

# text only
python llava_mask.py \
    --image_dir './vqa_samples_1k_2/images/' \
    --annotation_file './vqa_samples_1k_2/annotations.json' \
    --question_file './vqa_samples_1k_2/questions.json' \
    --output_dir './vqa_results_2_text_only' \
    --use_cross_attention \
    --cross_attention_image \
    --target_layers 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31