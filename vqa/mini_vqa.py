import os
import json
import shutil
import random

if __name__ == '__main__':
    
    num_samples = 1000

    annotation_file = "./data/v2_mscoco_val2014_annotations.json"
    question_file = "./data/v2_OpenEnded_mscoco_val2014_questions.json"
    data_dir = "./data/val2014/"

    dataset_folder = "./vqa_samples_1k_2"
    sample_annotation_file = f"{dataset_folder}/annotations.json"
    sample_question_file = f"{dataset_folder}/questions.json"
    sample_image_dir = f"{dataset_folder}/images/"

    try:
        os.mkdir(dataset_folder)
        print(f"Folder {dataset_folder} created successfully.")
    except FileExistsError:
        print(f"Folder {dataset_folder} already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        os.mkdir(f"{dataset_folder}/images")
        print(f"Folder {dataset_folder}/images created successfully.")
    except FileExistsError:
        print(f"Folder {dataset_folder}/images already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


    image_file_names = os.listdir(data_dir)
    print(len(image_file_names), image_file_names[0])
    image_samples = random.sample(image_file_names, num_samples)

    dataset = json.load(open(annotation_file, 'r'))
    questions = json.load(open(question_file, 'r'))

    sample_annotations = {
        "info": dataset["info"],
        "license": dataset["license"],
        "data_subtype": dataset["data_subtype"],
        "annotations": [],
        "data_type": dataset["data_type"],
    }
    sample_questions = {
        "info": questions["info"],
        "task_type": questions["task_type"],
        "data_type": questions["data_type"],
        "license": questions["license"],
        "data_subtype": questions["data_subtype"],
        "questions": [],
    }

    image_id_list = []
    for image in image_samples:
        img_id = int(image[:-4].split("_")[-1])
        shutil.copy(data_dir+image, sample_image_dir+image)
        image_id_list.append(img_id)

    question_id_list = []
    for annotation in dataset["annotations"]:
        if annotation["image_id"] in image_id_list:
            sample_annotations["annotations"].append(annotation)
            question_id_list.append(annotation["question_id"])

    for question in questions["questions"]:
        if question["question_id"] in question_id_list:
            sample_questions["questions"].append(question)

    print(len(question_id_list))

    with open(sample_annotation_file, "w") as file:
        json.dump(sample_annotations, file)
        print("sample annotations saved")

    with open(sample_question_file, "w") as file:
        json.dump(sample_questions, file)
        print("sample questions saved")



    # print(type(dataset), len(dataset), type(questions), len(questions))
    # for key in dataset.keys():
    #     print(key, len(dataset[key]), type(dataset[key]))
    # for key in questions.keys():
    #     print(key, len(questions[key]), type(questions[key]))

    # # print(dataset["annotations"][1])
    



    # sample_question_file = "./vqa_samples/questions.json"
    # questions = json.load(open(sample_question_file, 'r'))
    # print(questions.keys())
    # print(type(questions["questions"][0]["question_id"]))

    # sample_annotation_file = "./vqa_samples/annotations.json"
    # annotations = json.load(open(sample_annotation_file, 'r'))
    # print(annotations.keys())
    # print(annotations["annotations"][0])
