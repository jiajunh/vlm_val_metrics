import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


def construct_image_dict(image_dir):
    img_names = os.listdir(image_dir)
    img_dict = {}
    for img in img_names:
        idx = int(img[:-4].split('_')[-1])
        img_dict[idx] = os.path.join(image_dir, img)
    return img_dict

def construct_question_dict(question_file):
    questions = json.load(open(question_file, 'r'))
    question_dict = {}
    for q in questions["questions"]:
        question_dict[q["question_id"]] = q["question"]
    return question_dict

def construct_annotation_dict(annotation_file):
    annotations = json.load(open(annotation_file, 'r'))
    annotation_dict = {}
    for a in annotations["annotations"]:
        annotation_dict[a["question_id"]] = a["image_id"]
    return annotation_dict


class VQADataset(Dataset):
    def __init__(self, questions, question_ids, images, processor, model_config, pre_prompt, post_prompt):
        self.questions = questions
        self.question_ids = question_ids
        self.images = images
        self.processor = processor
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.model_config = model_config

    def __getitem__(self, index):
        prompt = f"USER: {self.pre_prompt}\n {self.questions[index]}. {self.post_prompt}\nASSISTANT:"
        image = Image.open(self.images[index]).convert("RGB")
        img_np = np.asarray(image)

        inputs = self.processor(images=img_np, text=prompt, return_tensors='pt')
        if self.model_config["device"]:
            inputs = inputs.to(self.model_config["device"], torch.float16)
        # print(inputs["input_ids"].shape, inputs["attention_mask"].shape, inputs["pixel_values"].shape)
        return (inputs, self.question_ids[index])

    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, question_ids, images, processor, model_config, pre_prompt="<image>", post_prompt="Answer the question using a single word or phrase."):
    vqa_dataset = VQADataset(questions=questions, 
                            question_ids=question_ids, 
                            images=images, 
                            processor=processor,
                            model_config=model_config, 
                            pre_prompt=pre_prompt, 
                            post_prompt=post_prompt)
    data_loader = DataLoader(vqa_dataset, batch_size=1, shuffle=False)
    return data_loader