# Script to prepare data for train, val and test. Data from VQA-Med 2019 is converted to the format that R2GenGPT model can use.

import os
import json
from tqdm import tqdm
from os.path import join as jp

path_orig = '/storage/homefs/st20f757/vqa/data/Tools/COCO-Regions_v1/'
path_output = '/storage/homefs/st20f757/vqa/locvqallm/data/coco_regions/processed'
# create output dir
os.makedirs(path_output, exist_ok=True)

subsets = ['train', 'val', 'test'] # using indexes for first digit of question ids

anns = {'train': [], 'val': [], 'test': []} # final annotations to be saved in json format
for subset in subsets:
    path_images = jp(path_orig, 'images', subset)
    path_qa = jp(path_orig, 'qa', f'{subset}_qa.json')
    # read qa.txt 
    with open(path_qa, 'r') as f:
        data = json.load(f)
    for l in tqdm(data, desc=f'Processing {subset}'):
        # get the image id
        image_id = l['image_name'].split('.')[0]
        # get the question
        question = l['question']
        # get alternative question with description of the region
        question_alt = l['question_alt']
        # get the mask coords
        mask_coords = l['mask_coords']
        # get the mask size
        mask_size = l['mask_size']
        # get the question type
        question_type = l['question_type']
        # get the question object
        question_object = l['question_object']
        # get the answer
        answer = l['answer']
        # get the image path
        image_path = subset + '/' + l['image_name']
        # append to anns
        q_id = l['question_id']
        anns[subset].append({   'id': image_id, 
                                'image_path': [image_path], 
                                'question': question, 
                                'question_alt': question_alt, 
                                'mask_coords': mask_coords,
                                'mask_size': mask_size,
                                'mask_type': 'rectangle',
                                'question_type': question_type,
                                'question_object': question_object,
                                'report': answer, 
                                'q_id': str(q_id) , 
                                'split': subset})
    
# save anns as json
with open(jp(path_output, 'anns.json'), 'w') as f:
    json.dump(anns, f)


