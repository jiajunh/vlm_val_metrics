import os
import json
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image

from utils import load_image, load_json_file, load_llava


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", type=str)
    parser.add_argument("--eval_mme", action="store_true", help="Enable MME evaluation")
    parser.add_argument("--eval_pope", action="store_true", help="Enable POPE evaluation")
    
    parser.add_argument("--mme_template_path", default="mme/eval_tool/Your_Results", type=str)
    parser.add_argument("--mme_output_path", default="results", type=str)
    parser.add_argument("--mme_data_path", default="mme/data", type=str)

    parser.add_argument("--pope_image_path", default="../pope_images", type=str)
    parser.add_argument("--pope_label_path", default="pope/coco_label/coco/coco_pope_adversarial.json", type=str)
    parser.add_argument("--pope_output_path", default="results")

    args = parser.parse_args()
    return args


def eval_mme(model, tokenizer, processor, args):
    args.mme_output_path = os.path.join(args.mme_output_path, args.model.split("/")[-1], "mme")
    os.makedirs(args.mme_output_path, exist_ok=True)

    generation_config = {
        "temperature": 0.0,
        "use_cache": True,
        "max_new_tokens": 8,
    }

    for filename in os.listdir(args.mme_template_path):
        with open(os.path.join(args.mme_template_path, filename), 'r') as fin, \
            open(os.path.join(args.mme_output_path, filename), 'w') as fout:
            lines = fin.read().splitlines()
            filename = filename.replace('.txt', '')

            for line in tqdm(lines, desc=filename):
                img, question, gt = line.strip().split('\t')
                img_path = os.path.join(args.mme_data_path, filename, img)
                assert os.path.exists(img_path), img_path

                img_np = load_image(img_path)
                prompt = f"USER: <image> \n{question} ASSISTANT:"

                with torch.no_grad():
                    inputs = processor(images=img_np, 
                                       text=prompt, 
                                       return_tensors="pt").to(model.device)
                    input_len = inputs["input_ids"].shape[1]
                    input_kwargs = {**inputs, **generation_config}
                    output = model.generate(**input_kwargs)
                    generated_ids = output[0][input_len:]

                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                # response = response.replace("\t", " ").replace("\n", " ")
                print(img, question, gt, response, sep='\t', file=fout)


def eval_pope(model, tokenizer, processor, args):
    args.pope_output_path = os.path.join(args.pope_output_path, args.model.split("/")[-1], "pope")
    os.makedirs(args.pope_output_path, exist_ok=True)
    args.pope_output_path = os.path.join(args.pope_output_path, "answers.json")
    
    generation_config = {
        "temperature": 0.0,
        "use_cache": True,
        "max_new_tokens": 8,
    }

    pope_label_data = [json.loads(q) for q in open(args.pope_label_path, 'r')]
    
    output_data = []
    for data in tqdm(pope_label_data, desc="pope_data"):
        question_id = data["question_id"]
        image = data["image"]
        question = data["text"]
        label = data["label"]

        img_path = os.path.join(args.pope_image_path, image)
        assert os.path.exists(img_path), img_path
        img_np = load_image(img_path)
        prompt = f"USER: <image> \n Answer yes or no. {question} ASSISTANT:"

        with torch.no_grad():
            inputs = processor(images=img_np, 
                               text=prompt, 
                               return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]
            input_kwargs = {**inputs, **generation_config}
            output = model.generate(**input_kwargs)
            generated_ids = output[0][input_len:]

        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        output_data.append({
            "question_id": question_id,
            "image": image,
            "question": question,
            "label": label,
            "answer": response
        })
    
    with open(args.pope_output_path, "w") as f:
        for entry in output_data:
            f.write(json.dumps(entry) + "\n")


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, processor = load_llava(
        model_name_or_path="llava-hf/llava-1.5-7b-hf",
        device_map="auto",
        quantization=True)
    
    print(args)
    
    if args.eval_mme:
        print("-" * 10 + "Evaluating MME" + "-" * 10)
        mme_start_time = time.time()
        eval_mme(model, tokenizer, processor, args)
        mme_end_time = time.time()
        print(f"MME Time: {(mme_end_time - mme_start_time):.4f}")
    
    if args.eval_pope:
        print("-" * 10 + "Evaluating POPE" + "-" * 10)
        pope_start_time = time.time()
        eval_pope(model, tokenizer, processor, args)
        pope_end_time = time.time()
        print(f"POPE Time: {(pope_end_time - pope_start_time):.4f}")