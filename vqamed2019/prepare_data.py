# Script to prepare data for train, val and test. Data from VQA-Med 2019 is converted to the format that R2GenGPT model can use.

import os
import json
from tqdm import tqdm
from os.path import join as jp

path_orig = '/storage/homefs/st20f757/vqa/locvqallm/data/vqa_med_2019/original'
path_output = '/storage/homefs/st20f757/vqa/locvqallm/data/vqa_med_2019/processed'

subsets = {'train': '1', 'val': '2', 'test': '3'} # using indexes for first digit of question ids

anns = {'train': [], 'val': [], 'test': []} # final annotations to be saved in json format
for subset, subset_index in subsets.items():
    path_subset = jp(path_orig, subset)
    path_images = jp(path_subset, 'images')
    path_qa = jp(path_subset, 'qa.txt')
    # read qa.txt 
    with open(path_qa) as f:
        lines = f.readlines()
    idx = 0
    for l in tqdm(lines, desc=f'Processing {subset}'):
        # remove the \n at the end of each line
        l = l.strip()
        # split the line into 3 parts using | as separator
        l = l.split('|')
        # get the image id
        image_id = l[0]
        # get the question
        question = l[1]
        # get the answer
        answer = l[2]
        # get the image path
        image_path = jp(subset, 'images', image_id + '.jpg')
        # append to anns
        q_id = subset_index + str(idx).zfill(5)
        anns[subset].append({'id': image_id, 'image_path': [image_path], 'question': question, 'report': answer, 'q_id': q_id , 'split': subset})
        idx += 1
    
# save anns as json
with open(jp(path_output, 'anns.json'), 'w') as f:
    json.dump(anns, f)


