import os
import json
import time
import math
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F
from types import MethodType


from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import load_image, load_json_file

GLOBAL_FIRST_IMG_IDX = None
POSITION_EMBEDDING_COS = None
POSITION_EMBEDDING_SIN = None


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", type=str)
    parser.add_argument("--eval_mme", action="store_true", help="Enable MME evaluation")
    parser.add_argument("--eval_pope", action="store_true", help="Enable POPE evaluation")
    
    parser.add_argument("--mme_template_path", default="mme/eval_tool/Your_Results", type=str)
    parser.add_argument("--mme_output_path", default="results/cross_attention", type=str)
    parser.add_argument("--mme_data_path", default="mme/data", type=str)

    parser.add_argument("--pope_image_path", default="pope_images", type=str)
    parser.add_argument("--pope_label_path", default="pope/coco_label/coco/coco_pope_adversarial.json", type=str)
    parser.add_argument("--pope_output_path", default="results/cross_attention")

    # model_info
    parser.add_argument('--use_cross_attention', action='store_true')
    parser.add_argument('--cross_attention_image', action='store_true')
    parser.add_argument('--target_layers', nargs="*", type=int)
    parser.add_argument('--layer_type', choices=["1", "3c"], default="1")

    args = parser.parse_args()
    return args



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


def eval_mme(args, target_layer):
    # args.mme_output_path = os.path.join(args.mme_output_path, "mme")
    subset = "ca_image" if args.cross_attention_image else "ca_text"
    layer_str = "_".join(map(str, target_layer))

    output_file = os.path.join(args.mme_output_path, subset, "mme", "baseline")
    if target_layer:
        output_file = os.path.join(args.mme_output_path, subset, "mme", layer_str)
    
    os.makedirs(output_file, exist_ok=True)

    generation_config = {
        "temperature": 0.0,
        "use_cache": True,
        "max_new_tokens": 8,
    }

    print("-"*30+f"target layer: {target_layer}"+"-"*30)
    print("-"*30+"Load model"+"-"*30)
    model, tokenizer, processor = load_llava(
        model_name_or_path="llava-hf/llava-1.5-7b-hf",
        device_map="auto",
        quantization= True)

    if target_layer:
        print("-"*30+"Bind new forward function"+"-"*30)
        load_cross_attention_module(model, 
                                    target_layers=target_layer, 
                                    ca_img=args.cross_attention_image,
                                    use_cross_attention=args.use_cross_attention)
    else:
        print("-"*30+"Use original model"+"-"*30)

    for filename in os.listdir(args.mme_template_path):
        with open(os.path.join(args.mme_template_path, filename), 'r') as fin, \
            open(os.path.join(output_file, filename), 'w') as fout:
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
                    
                    input_kwargs = {**inputs, **generation_config}
                    
                    input_len = inputs["input_ids"].shape[1]
                    output = model.generate(**input_kwargs)
                    generated_ids = output[0][input_len:]

                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(img, question, gt, response, sep='\t', file=fout)


def eval_pope(args, target_layer):
    subset = "ca_image" if args.cross_attention_image else "ca_text"
    layer_str = "_".join(map(str, target_layer))

    output_file = os.path.join(args.pope_output_path, subset, "pope", "baseline")
    if target_layer:
        output_file = os.path.join(args.pope_output_path, subset, "pope", layer_str)
    os.makedirs(output_file, exist_ok=True)

    output_file = os.path.join(output_file, "answers.json")

    generation_config = {
        "temperature": 0.0,
        "use_cache": True,
        "max_new_tokens": 8,
    }

    print("-"*30+f"target layer: {target_layer}"+"-"*30)
    print("-"*30+"Load model"+"-"*30)
    model, tokenizer, processor = load_llava(
        model_name_or_path="llava-hf/llava-1.5-7b-hf",
        device_map="auto",
        quantization= True)

    if target_layer:
        print("-"*30+"Bind new forward function"+"-"*30)
        load_cross_attention_module(model, 
                                    target_layers=target_layer, 
                                    ca_img=args.cross_attention_image,
                                    use_cross_attention=args.use_cross_attention)
    else:
        print("-"*30+"Use original model"+"-"*30)
    
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
        output_data.append(data_item)
    
    with open(output_file, "w") as f:
        for entry in output_data:
            f.write(json.dumps(entry) + "\n")


if __name__ == '__main__':
    args = parse_args()
    print(args)

    subset = "ca_image" if args.cross_attention_image else "ca_text"

    model, tokenizer, processor = load_llava(
        model_name_or_path="llava-hf/llava-1.5-7b-hf",
        device_map="auto",
        quantization= True)
    device = model.device
    
    longest_prompt = f"USER: <image> \n ASSISTANT:"
    init(model, longest_prompt, 2048)

    if args.eval_mme:
        print("-" * 10 + "Evaluating MME" + "-" * 10)
        mme_start_time = time.time()

        if not args.target_layers:
            eval_mme(args=args, target_layer=[])
        else:
            if args.layer_type == "1":
                # Every time change one layer
                for target_layer in args.target_layers:
                    eval_mme(args=args, target_layer=[target_layer])
            elif args.layer_type == "3c":
                # 3 consececutive layers
                for target_layer in args.target_layers:
                    if target_layer < 30:
                        eval_mme(args=args, target_layer=[target_layer, target_layer+1, target_layer+2])

        mme_end_time = time.time()
        print(f"MME Time: {(mme_end_time - mme_start_time):.4f}")
    

    if args.eval_pope:
        print("-" * 10 + "Evaluating POPE" + "-" * 10)
        pope_start_time = time.time()

        if not args.target_layers:
            eval_pope(args=args, target_layer=[])
        else:
            if args.layer_type == "1":
                # Every time change one layer
                for target_layer in args.target_layers:
                    eval_pope(args=args, target_layer=[target_layer])
            elif args.layer_type == "3c":
                # 3 consececutive layers
                for target_layer in args.target_layers:
                    if target_layer < 30:
                        eval_pope(args=args, target_layer=[target_layer, target_layer+1, target_layer+2])


        pope_end_time = time.time()
        print(f"POPE Time: {(pope_end_time - pope_start_time):.4f}")