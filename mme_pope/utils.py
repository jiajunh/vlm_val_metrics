import os
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig


def load_json_file(file_path):
    if file_path.split(".")[-1] == "jsonl":
        json_data = [json.loads(s) for s in open(file_path)]
    else:
        json_data = json.load(open(file_path))
    return json_data


def load_llava(model_name_or_path,
               device_map="auto",
               quantization=True):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        quantization_config=quantization_config)
    return model, tokenizer, processor


def load_image(image_path):
    image = Image.open(image_path)
    img_np = np.asarray(image)
    if len(img_np.shape) == 2:
        img_np = np.stack((img_np,)*3, axis=-1)
    return img_np
