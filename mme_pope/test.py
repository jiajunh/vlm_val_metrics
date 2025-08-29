from utils import load_json_file
import json

answers = [json.loads(q) for q in open("pope/coco_label/coco/coco_pope_adversarial.json", 'r')]
print(type(answers), len(answers))
print(answers[0])