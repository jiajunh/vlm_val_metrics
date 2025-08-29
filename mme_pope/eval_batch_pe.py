import os
import json
import time
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F

from utils import load_image, load_json_file, load_llava


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", type=str)
    parser.add_argument("--eval_mme", action="store_true", help="Enable MME evaluation")
    parser.add_argument("--eval_pope", action="store_true", help="Enable POPE evaluation")

    parser.add_argument("--prob_threshold", default=0.4, type=float)
    parser.add_argument("--failure_patch_num", default=20, type=int)
    parser.add_argument("--img_token_id", default=32000, type=int)
    parser.add_argument("--image_token_size", default=576, type=int)
    
    parser.add_argument("--mme_template_path", default="mme/eval_tool/Your_Results", type=str)
    parser.add_argument("--mme_output_path", default="results/patch_elimination", type=str)
    parser.add_argument("--mme_data_path", default="mme/data", type=str)

    parser.add_argument("--pope_image_path", default="../pope_images", type=str)
    parser.add_argument("--pope_label_path", default="pope/coco_label/coco/coco_pope_adversarial.json", type=str)
    parser.add_argument("--pope_output_path", default="results/patch_elimination")

    args = parser.parse_args()
    return args


def get_first_image_token_idx(inputs, img_token_id=32000):
    first_img_token_idx = (inputs["input_ids"][0]==img_token_id).nonzero(as_tuple=True)[0][0].detach().cpu().numpy()
    return first_img_token_idx


def generate_objects(model, processor, img_np):
    prompt = (
        "USER: <image>\n"
        "Please list all distinct, singular object names clearly visible in the image. "
        "Include people or animals if present. Do not include colors, materials, or adjectives. "
        "Only name the actual objects, separated by commas.\nASSISTANT:"
    )
    inputs = processor(images=img_np, text=prompt, return_tensors='pt').to(model.device, torch.float16)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=32)
    generated_ids = output[:,input_len:]
    generated_sequences = processor.batch_decode(generated_ids,
                                                 skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)

    obj_set = set([obj.strip().lower() for obj in generated_sequences[0].split(",")])
    return list(obj_set)


def batch_iterative_elimination(model, obj_list, inputs, 
                                threshold=0.5, 
                                failure_patch_num=50, 
                                img_token_id=32000, 
                                image_token_size=576):
    logits_generate_config = {
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "max_new_tokens": 1,
    }
    first_img_token_idx = (inputs["input_ids"][0]==img_token_id).nonzero(as_tuple=True)[0][0].detach().cpu().numpy()
    projection_layer = model.language_model.lm_head

    obj_token_ids = list(set([ids for obj in obj_list for ids in tokenizer(obj)["input_ids"][1:]]))
    patch_list = list(range(image_token_size))
    num_patch = 0
    prob_list = []
    
    failure_patch_list = patch_list
    failure_prob_list = []

    for i in range(10):
        num_patch = len(patch_list)
        mask_list = [first_img_token_idx+idx for idx in patch_list]
        inputs["attention_mask"][0,first_img_token_idx:first_img_token_idx+image_token_size] = 0
        inputs["attention_mask"][0,mask_list] = 1
        input_kwargs = {**inputs, **logits_generate_config}

        with torch.inference_mode():
            outputs = model.generate(**input_kwargs)
            hidden_states = torch.stack(outputs.hidden_states[0])[:,:,first_img_token_idx:first_img_token_idx+image_token_size,:]
            logits = projection_layer(hidden_states).cpu()
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            internal_conf_heatmap = np.max(prob[:,:,:, obj_token_ids], axis=-1).squeeze()
        probs = np.max(internal_conf_heatmap, axis=0)
        
        if not prob_list:
            prob_list = probs
            failure_prob_list = probs
            
        probs_mask = np.zeros_like(probs)
        probs_mask[patch_list] = probs[patch_list] 

        filter_patch_length = sum(probs_mask > threshold)
        
        indices = np.argsort(probs_mask)[::-1]
        patch_list = indices[:filter_patch_length].tolist()
        prob_list = probs[patch_list].tolist()

        if filter_patch_length >= failure_patch_num:
            failure_patch_list = patch_list
            failure_prob_list = prob_list

        if len(patch_list) == 0:
            print(False, len(failure_patch_list))
            return failure_patch_list, failure_prob_list, False

        if len(patch_list) == num_patch:
            break

    print(True, len(patch_list))
    return patch_list, prob_list, True


def eval_mme(model, tokenizer, processor, args):
    args.mme_output_path = os.path.join(args.mme_output_path, "mme")
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

                # Patch Elimination
                obj_list = generate_objects(model, processor, img_np)
                prompt_pe = "USER: <image>\n Describe the image. \nASSISTANT:"
                inputs_pe = processor(images=img_np, 
                                      text=prompt_pe, 
                                      return_tensors="pt").to(model.device)
                patch_list, prob_list, converged = batch_iterative_elimination(
                    model, obj_list, inputs_pe, 
                    threshold=args.prob_threshold, 
                    failure_patch_num=args.failure_patch_num, 
                    img_token_id=args.img_token_id, 
                    image_token_size=args.image_token_size)

                with torch.no_grad():
                    inputs = processor(images=img_np, 
                                       text=prompt, 
                                       return_tensors="pt").to(model.device)
                    
                    first_image_token_idx = get_first_image_token_idx(inputs, img_token_id=args.img_token_id)

                    mask_list = [first_image_token_idx+idx for idx in patch_list]
                    inputs["attention_mask"][0,first_image_token_idx:first_image_token_idx+args.image_token_size] = 0
                    inputs["attention_mask"][0,mask_list] = 1
                    input_kwargs = {**inputs, **generation_config}
                    
                    input_len = inputs["input_ids"].shape[1]
                    output = model.generate(**input_kwargs)
                    generated_ids = output[0][input_len:]

                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                # response = response.replace("\t", " ").replace("\n", " ")
                print(img, question, gt, response, sep='\t', file=fout)


def eval_pope(model, tokenizer, processor, args):
    args.pope_output_path = os.path.join(args.pope_output_path, "pope")
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

        # Patch Elimination
        obj_list = generate_objects(model, processor, img_np)
        prompt_pe = "USER: <image>\n Describe the image. \nASSISTANT:"
        inputs_pe = processor(images=img_np, 
                              text=prompt_pe, 
                              return_tensors="pt").to(model.device)
        patch_list, prob_list, converged = batch_iterative_elimination(
                    model, obj_list, inputs_pe, 
                    threshold=args.prob_threshold, 
                    failure_patch_num=args.failure_patch_num, 
                    img_token_id=args.img_token_id, 
                    image_token_size=args.image_token_size)

        with torch.no_grad():
            inputs = processor(images=img_np, 
                               text=prompt, 
                               return_tensors="pt").to(model.device)
            
            first_image_token_idx = get_first_image_token_idx(inputs, img_token_id=args.img_token_id)

            mask_list = [first_image_token_idx+idx for idx in patch_list]
            inputs["attention_mask"][0,first_image_token_idx:first_image_token_idx+args.image_token_size] = 0
            inputs["attention_mask"][0,mask_list] = 1
            input_kwargs = {**inputs, **generation_config}
            
            input_len = inputs["input_ids"].shape[1]
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