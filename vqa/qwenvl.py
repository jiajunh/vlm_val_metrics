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

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import BitsAndBytesConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb as hf_apply_mrope, 
    repeat_kv
)

from load_vqa_dataset import create_data_loader, construct_image_dict, construct_question_dict, construct_annotation_dict

GLOBAL_FIRST_IMG_IDX = 15

def load_model(model_name_or_path, device_map="auto"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path,
                                                device_map=device_map,
                                                attn_implementation="eager",
                                                quantization_config=quantization_config)
    return model, tokenizer, processor


def make_cross_attention_forward(layer_idx, rotary_emb, ca_img=True, use_cross_attention=True):
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

        # position_ids: [3, batch, seq_len / 1]
        # attention_mask: [1,1, seq, seq] / [1,1,1,seq]
        # position_embeddings: tuple (cos, sin)
        #                 cos: [3, 1, sqe / 1, 128]

        if position_ids.shape[-1] > 1:
            global GLOBAL_FIRST_IMG_IDX
            self.img_start_idx = (position_ids[0] == GLOBAL_FIRST_IMG_IDX).nonzero()[0,1] - 1
            self.img_end_idx = (position_ids[0] == GLOBAL_FIRST_IMG_IDX).nonzero()[-1,1] + 1
            self.img_patch_num = self.img_end_idx - self.img_start_idx + 1

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
                modified_position_ids[:, :, self.img_start_idx:self.img_end_idx] -= self.img_start_idx
                modified_position_ids[:, :, self.img_end_idx:] -= self.img_patch_num
            else:
                modified_position_ids[:,:,-1] = modified_position_ids[:,:,-1] - self.img_patch_num

            cos, sin = rotary_emb(hidden_states, position_ids=modified_position_ids)

            if ca_img:
                modified_attention_mask[:,:,:,self.img_start_idx:self.img_end_idx+1] = -float("inf")
                inverse_modified_attention_mask[:,:,:,0:self.img_start_idx] = -float("inf")
                inverse_modified_attention_mask[:,:,:,self.img_end_idx+1:] = -float("inf")
            else:
                modified_attention_mask[:,:,:,0:self.img_start_idx] = -float("inf")
                modified_attention_mask[:,:,:,self.img_end_idx+1:] = -float("inf")
                inverse_modified_attention_mask[:,:,:,self.img_start_idx:self.img_end_idx+1] = -float("inf")

        cos = cos.to(device)
        sin = sin.to(device)

        modified_position_embeddings = (cos, sin)
        origin_hidden_state = hidden_states

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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
            q = self.self_attn.q_proj(hidden_states).to(device)
            # q = self.self_attn.q_proj(origin_hidden_state).to(device)

            batch_size, seq_len, hidden_dim = q.shape
            head_dim = self.self_attn.head_dim
            num_heads = hidden_dim // head_dim


            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            # k = ca_k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            # v = ca_v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)


            # q, k = apply_rotary_pos_emb(q, q, cos, sin)
            q, _ = hf_apply_mrope(
                q, q, cos, sin, self.self_attn.rope_scaling["mrope_section"]
            )

            k = repeat_kv(past_key_value.key_cache[layer_idx], self.self_attn.num_key_value_groups)
            v = repeat_kv(past_key_value.value_cache[layer_idx], self.self_attn.num_key_value_groups)
    
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
            # outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
            
    return new_module_forward


def load_cross_attention_module(model, rotary_emb, target_layers=[-1],
                                ca_img=False, use_cross_attention=True):
    for layer_idx in target_layers:
        decoder_layer = model.language_model.layers[layer_idx]
        decoder_layer.forward = MethodType(
            make_cross_attention_forward(layer_idx, rotary_emb, ca_img, use_cross_attention), 
            decoder_layer)

def run_generations(args, target_layer, data_loader):
    output_responses = []
    subset = "qwenvl_ca_image" if args.cross_attention_image else "qwenvl_ca_text"
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
    model, tokenizer, processor = load_model(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="auto")
    data_loader.processor = processor
    rotary_emb = model.language_model.rotary_emb

    if target_layer:
        print("-"*30+"Bind new forward function"+"-"*30)
        load_cross_attention_module(model, 
                                    rotary_emb = rotary_emb,
                                    target_layers=target_layer,
                                    ca_img=args.cross_attention_image,
                                    use_cross_attention=args.use_cross_attention)
    else:
        print("-"*30+"Use original model"+"-"*30)

    for inputs, question_ids in tqdm(data_loader, total=len(data_loader)):
        output = model.generate(**inputs, **generate_config)
        output_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        response_answer = output_text.split("assistant")[-1].strip()
        output_responses.append({
            "answer": response_answer,
            "question_id": int(question_ids)
        })

    with open(output_file, "w") as file:
        json.dump(output_responses, file)
        print(f"{output_file} saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # vqa settings
    parser.add_argument('--image_dir', type=str, default='./vqa_samples_1k_2/images/')
    parser.add_argument('--annotation_file', type=str, default='./vqa_samples_1k_2/annotations.json')
    parser.add_argument('--question_file', type=str, default='./vqa_samples_1k_2/questions.json')
    
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

    subset = "qwenvl_ca_image" if args.cross_attention_image else "qwenvl_ca_text"
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
    model, tokenizer, processor = load_model(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        device_map="auto")

    device = model.device
    print(f"device: {device}")


    # Constructor dataloader
    data_loader = create_data_loader(
        questions=questions, 
        question_ids=question_ids, 
        images=image_paths, 
        processor=processor,
        model_config={"device": device, "model_type": "qwenvl"}, 
        pre_prompt=None, 
        post_prompt=args.instruction,
    )
    
    
    
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
    