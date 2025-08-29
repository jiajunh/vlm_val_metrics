import os
import math
import time
import requests
import json
import torch
import random
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from types import MethodType

from PIL import Image
from torch.nn import functional as F

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from load_vqa_dataset import create_data_loader, construct_image_dict, construct_question_dict, construct_annotation_dict


GLOBAL_FIRST_IMG_IDX = None
POSITION_EMBEDDING_COS = None
POSITION_EMBEDDING_SIN = None


def load_llava(model_name_or_path,
               device_map="auto",
               quantization = True):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path,
                                                          device_map=device_map,
                                                          attn_implementation="eager",
                                                          quantization_config=quantization_config)
    return model, tokenizer, processor


def get_first_image_token_idx(inputs, img_id=32000):
    mask = (inputs["input_ids"] == img_id)
    indices = torch.nonzero(mask)
    return indices[0,1]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def make_cross_attention_forward(layer_idx, ca_img=True, use_cross_attention=True):
    patch_num = 576
    def new_module_forward(
        self, hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        cache_position = None,
        position_embeddings = None, 
        **kwargs,
    ):

        # Original self-attention layernorm
        device = hidden_states.device
        cos, sin = position_embeddings
        
        residual = hidden_states  
        hidden_states = self.input_layernorm(hidden_states)

        # # if attention_mask is None

        modified_attention_mask = attention_mask.clone().to(device)
        inverse_modified_attention_mask = attention_mask.clone().to(device)
        modified_position_ids = position_ids.clone().to(device)
        modified_position_embeddings = None

        if use_cross_attention:
            if modified_position_ids.shape[-1] > 1:
                modified_position_ids[:,GLOBAL_FIRST_IMG_IDX:GLOBAL_FIRST_IMG_IDX+patch_num] = range_tensor = torch.arange(0, patch_num)
                modified_position_ids[:,GLOBAL_FIRST_IMG_IDX+patch_num:] = modified_position_ids[:,GLOBAL_FIRST_IMG_IDX+patch_num:]-patch_num
            else:
                modified_position_ids[:,-1] = modified_position_ids[:,-1]-patch_num

            cos = POSITION_EMBEDDING_COS[:, modified_position_ids[-1], :].to(device)
            sin = POSITION_EMBEDDING_SIN[:, modified_position_ids[-1], :].to(device)

            if ca_img:
                modified_attention_mask[:,:,:,GLOBAL_FIRST_IMG_IDX:GLOBAL_FIRST_IMG_IDX+patch_num] = -float("inf") #-65504.0
                inverse_modified_attention_mask[:,:,:,0:GLOBAL_FIRST_IMG_IDX] = -float("inf") #-65504.0
                inverse_modified_attention_mask[:,:,:,GLOBAL_FIRST_IMG_IDX+patch_num:] = -float("inf") #-65504.0
            else:
                modified_attention_mask[:,:,:,0:GLOBAL_FIRST_IMG_IDX] = -float("inf") #-65504.0
                modified_attention_mask[:,:,:,GLOBAL_FIRST_IMG_IDX+patch_num:] = -float("inf") #-65504.0
                inverse_modified_attention_mask[:,:,:,GLOBAL_FIRST_IMG_IDX:GLOBAL_FIRST_IMG_IDX+patch_num] = -float("inf")

        cos = cos.to(device)
        sin = sin.to(device)

        modified_position_embeddings = (cos, sin)
            
        origin_hidden_state = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=modified_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=modified_position_embeddings,
            **kwargs,
        )
        
        if use_cross_attention:
            # cross atten
            # hidden_states = self.input_layernorm(hidden_states)

            q = self.self_attn.q_proj(hidden_states).to(device)
            # q = self.self_attn.q_proj(origin_hidden_state).to(device)

            batch_size, seq_len, hidden_dim = q.shape
            head_dim = self.self_attn.head_dim
            num_heads = hidden_dim // head_dim

            # # --------------------------------------------------------------
            # # if not use self attention
            # ca_k = self.self_attn.k_proj(hidden_states)
            # ca_v = self.self_attn.v_proj(hidden_states)

            # q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            # k = ca_k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            # v = ca_v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            # q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            # if past_key_value is not None:
            #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            #     k, v = past_key_value.update(k, v, layer_idx, cache_kwargs)

            # --------------------------------------------------------------
            # If using self attention
            # ca_k = self.self_attn.k_proj(origin_hidden_state)
            # ca_v = self.self_attn.v_proj(origin_hidden_state)

            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            # k = ca_k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            # v = ca_v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, q, cos, sin)
            k = past_key_value.key_cache[layer_idx]
            v = past_key_value.value_cache[layer_idx]
    
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_scores += inverse_modified_attention_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
            hidden_states = self.self_attn.o_proj(attn_output)


        residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # output
        outputs = (hidden_states,)
        # outputs = (cross_hidden_states,)
        if output_attentions:
            if use_cross_attention:
                outputs += (attn_weights,)
            else:
                outputs += (self_attn_weights,)
        return outputs
    return new_module_forward


def load_cross_attention_module(model, target_layers=[-1], 
                                ca_img=False, use_cross_attention=True):
    for layer_idx in target_layers:
        decoder_layer = model.language_model.model.layers[layer_idx]
        decoder_layer.forward = MethodType(make_cross_attention_forward(layer_idx, ca_img, use_cross_attention), decoder_layer)
    

def init(model, prompt, max_new_tokens):
    inputs = processor(text=prompt, return_tensors='pt').to(model.device, torch.float16)
    first_img_idx = get_first_image_token_idx(inputs)
    global GLOBAL_FIRST_IMG_IDX
    GLOBAL_FIRST_IMG_IDX = first_img_idx
    print(f"GLOBAL_FIRST_IMG_IDX: {GLOBAL_FIRST_IMG_IDX}")

    rope = model.language_model.model.rotary_emb

    seq_len = inputs['input_ids'].shape[-1] + 575 + max_new_tokens
    print(f"longest seq_len: {seq_len}")
    x = torch.ones(1, seq_len, 128, device=model.device, dtype=torch.float16)
    position_ids = torch.arange(seq_len).unsqueeze(0).to(model.device)
    cos, sin = rope(x, position_ids=position_ids)
    global POSITION_EMBEDDING_COS
    global POSITION_EMBEDDING_SIN
    POSITION_EMBEDDING_COS = cos
    POSITION_EMBEDDING_SIN = sin


def run_generations(args, target_layer, data_loader):
    output_responses = []
    subset = "ca_image" if args.cross_attention_image else "ca_text"
    layer_str = "_".join(map(str, target_layer))
    output_file = os.path.join(args.output_dir, subset, f"baseline.json")
    if target_layer:
        output_file = os.path.join(args.output_dir, subset, f"{layer_str}.json")

    generate_config = {
        "return_dict_in_generate": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "max_new_tokens": args.max_new_tokens,
    }

    print("-"*30+f"target layer: {target_layer}"+"-"*30)
    print("-"*30+"Load model"+"-"*30)
    model, tokenizer, processor = load_llava(
        model_name_or_path="llava-hf/llava-1.5-7b-hf",
        device_map="auto",
        quantization= True)
    data_loader.processor = processor

    if target_layer:
        print("-"*30+"Bind new forward function"+"-"*30)
        load_cross_attention_module(model, 
                                    target_layers=target_layer, 
                                    ca_img=args.cross_attention_image,
                                    use_cross_attention=args.use_cross_attention)
    else:
        print("-"*30+"Use original model"+"-"*30)

    for inputs, question_ids in tqdm(data_loader, total=len(data_loader)):
        inputs["input_ids"] = inputs["input_ids"].squeeze(dim=0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(dim=0)
        inputs["pixel_values"] = inputs["pixel_values"].squeeze(dim=0)
        output = model.generate(**inputs, **generate_config)
        output_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        response_answer = output_text.split("ASSISTANT: ")[-1].strip()
        output_responses.append({
            "answer": response_answer,
            "question_id": int(question_ids.item())
        })
        # break

    with open(output_file, "w") as file:
        json.dump(output_responses, file)
        print(f"{output_file} saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # vqa settings
    parser.add_argument('--image_dir', type=str, default='./vqa_samples_1k/images/')
    parser.add_argument('--annotation_file', type=str, default='./vqa_samples_1k/annotations.json')
    parser.add_argument('--question_file', type=str, default='./vqa_samples_1k/questions.json')
    
    # model_info
    parser.add_argument('--use_cross_attention', action='store_true')
    parser.add_argument('--cross_attention_image', action='store_true')
    parser.add_argument('--target_layers', nargs="*", type=int)
    parser.add_argument('--layer_type', choices=["1", "3c"], default="1")

    # 
    parser.add_argument('--instruction', type=str, default='Answer the question using a single word or phrase.')
    parser.add_argument('--output_dir', type=str, default='./vqa_results')
    parser.add_argument('--max_new_tokens', type=int, default=16)

    args = parser.parse_args()
    print(args)

    subset = "ca_image" if args.cross_attention_image else "ca_text"
    os.makedirs(os.path.join(args.output_dir, subset), exist_ok=True)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    image_dict = construct_image_dict(args.image_dir)
    question_dict = construct_question_dict(args.question_file)
    annotation_dict = construct_annotation_dict(args.annotation_file)

    questions = []
    question_ids = []
    image_paths = []
    for q_id, q in question_dict.items():
        question_ids.append(q_id)
        questions.append(q)
        img_id = annotation_dict[q_id]
        image_paths.append(image_dict[img_id])

    print("-"*30+"Load model"+"-"*30)
    model, tokenizer, processor = load_llava(
        model_name_or_path="llava-hf/llava-1.5-7b-hf",
        device_map="auto",
        quantization= True)

    device = model.device
    print(f"device: {device}")

    pre_prompt = "<image>"
    post_prompt = args.instruction

    data_loader = create_data_loader(
        questions=questions, 
        question_ids=question_ids, 
        images=image_paths, 
        processor=processor,
        model_config={"device": device}, 
        pre_prompt="<image>", 
        post_prompt=args.instruction,
    )

    # Find longest input
    print("-"*30+"Finding longest prompt"+"-"*30)
    max_length = 0
    max_length_question_id = -1
    max_length_idx = 0
    for i, (inputs, question_ids) in enumerate(data_loader):
        length = inputs["input_ids"].shape[-1]
        if length > max_length:
            max_length = length
            max_length_question_id = int(question_ids.item())
            max_length_idx = i
    print(f"max_length_idx: {max_length_idx}, max_length: {max_length}, max_length_question_id: {max_length_question_id}")

    # init
    print("-"*30+"initiate model"+"-"*30)
    longest_prompt = f"USER: <image>\n {question_dict[max_length_question_id]}. {args.instruction}\nASSISTANT:"
    init(model, longest_prompt, args.max_new_tokens)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    if not args.target_layers:
        run_generations(args=args, target_layer=[], data_loader=data_loader)
    else:
        if args.layer_type == "1":
            # Every time change one layer
            for target_layer in args.target_layers:
                run_generations(args=args, target_layer=[target_layer], data_loader=data_loader)
        elif args.layer_type == "3c":
            # 3 consececutive layers
                for target_layer in args.target_layers:
                    if target_layer < 30:
                        run_generations(args=args, target_layer=[target_layer, target_layer+1, target_layer+2], data_loader=data_loader)
