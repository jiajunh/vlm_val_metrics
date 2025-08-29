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
    parser.add_argument("--save_patches", action="store_true")

    parser.add_argument("--rate", default=0.5, type=float)
    parser.add_argument("--iters", default=4, type=int)
    parser.add_argument("--img_token_id", default=32000, type=int)
    parser.add_argument("--image_token_size", default=576, type=int)
    parser.add_argument("--remaining_patch_num", default=-1, type=int)
    parser.add_argument("--r_duplicate", default=64, type=int)
    
    parser.add_argument("--mme_template_path", default="mme/eval_tool/Your_Results", type=str)
    parser.add_argument("--mme_output_path", default="results/percentage_pe", type=str)
    parser.add_argument("--mme_data_path", default="mme/data", type=str)

    parser.add_argument("--pope_image_path", default="../pope_images", type=str)
    parser.add_argument("--pope_label_path", default="pope/coco_label/coco/coco_pope_adversarial.json", type=str)
    parser.add_argument("--pope_output_path", default="results/percentage_pe")

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


def diverse_token(image_features, patch_list, num_div_token=64, r=64):
    num_patch = image_features.shape[1]
    feature_norm = torch.nn.functional.normalize(image_features, p=2, dim=2).detach().cpu()

    remaining_idx = [i for i in range(num_patch) if i not in set(patch_list)]
    remaining_idx = torch.tensor(remaining_idx)

    while r > 0:
        remaining = feature_norm[0,remaining_idx,:]
        
        a = remaining[0::2]
        b = remaining[1::2]
        
        score = a @ b.transpose(-1, -2)        
        score = score.max(dim=-1).values
        diverse_idx = score.argsort(dim=-1,descending=True)[r:]
        
        remaining_idx = torch.cat((remaining_idx[::2][diverse_idx], remaining_idx[1::2]), dim=-1)
        r = min(r, remaining_idx.shape[0] - num_div_token)

    remaining_idx = remaining_idx.tolist()
    return remaining_idx


def percentage_iterative_elimination(model, obj_list, img_np, 
                                     rate=0.5, iters=4,
                                     patch_num=-1, r_duplicate=64,
                                     img_token_id=32000, 
                                     image_token_size=576):
    prompt = "USER: <image>\n Describe the image. \nASSISTANT:"
    inputs = processor(images=img_np, 
                       text=prompt, 
                       return_tensors="pt").to(model.device, torch.float16)

    image_features = model.get_image_features(
        pixel_values=inputs.pixel_values,
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy
    )
    inputs.pop("pixel_values")
    logits_generate_config = {
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "max_new_tokens": 1,
    }
    
    projection_layer = model.language_model.lm_head

    remain_patches_num = image_token_size
    obj_token_ids = list(set([ids for obj in obj_list for ids in tokenizer(obj)["input_ids"][1:]]))
    patch_list = list(range(image_token_size))
    prob_list = []
    
    for i in range(iters):
        remain_patches_num = int(remain_patches_num * rate)

        reduced_image_feature = image_features[:, patch_list, :]
        img_token_start_idx = (inputs["input_ids"] == img_token_id).nonzero()[0][1].detach().cpu().numpy()
        img_token_end_idx = img_token_start_idx + image_token_size
        input_ids = torch.cat(
            (inputs["input_ids"][:, :img_token_start_idx+len(patch_list)],
             inputs["input_ids"][:, img_token_end_idx+1:]), dim=1)
        attention_mask = inputs["attention_mask"][:, :input_ids.shape[-1]]
        inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds[:, img_token_start_idx:img_token_start_idx + len(patch_list), :] = reduced_image_feature
        
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        inputs["inputs_embeds"] = inputs_embeds
        
        input_kwargs = {**inputs, **logits_generate_config}

        with torch.inference_mode():
            outputs = model.generate(**input_kwargs)
            hidden_states = torch.stack(outputs.hidden_states[0])[:,:,img_token_start_idx:img_token_start_idx+len(patch_list),:]
            logits = projection_layer(hidden_states).cpu()
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            internal_conf_heatmap = np.max(prob[:,:,:, obj_token_ids], axis=-1).squeeze()
        
        probs = np.max(internal_conf_heatmap, axis=0)
        index_list = np.argsort(-probs)[:remain_patches_num].tolist()
        patch_list = np.sort(np.array(patch_list)[index_list]).tolist()
    
    if patch_num > len(patch_list):
        div_patch_list = diverse_token(image_features, patch_list, num_div_token=patch_num-len(patch_list), r=r_duplicate)
        patch_list = patch_list + div_patch_list
        patch_list.sort()
        
    return patch_list


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
                
                patch_list = percentage_iterative_elimination(
                    model, obj_list, img_np, 
                    rate=args.rate, 
                    iters=args.iters, 
                    patch_num=args.remaining_patch_num,
                    r_duplicate=args.r_duplicate,
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
                if args.save_patches:
                    print(img, question, gt, response, patch_list, sep='\t', file=fout)
                else:
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

        patch_list = percentage_iterative_elimination(
            model, obj_list, img_np, 
            rate=args.rate, 
            iters=args.iters, 
            patch_num=args.remaining_patch_num,
            r_duplicate=args.r_duplicate,
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
        
        data_item = {
            "question_id": question_id,
            "image": image,
            "question": question,
            "label": label,
            "answer": response,
        }
        if args.save_patches:
            data_item["patches"] = patch_list
        output_data.append(data_item)
    
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