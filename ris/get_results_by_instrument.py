import json
import os
import numpy as np
from tqdm import tqdm
from os.path import join as jp
from PIL import Image
import matplotlib.pyplot as plt



version = 'v1'
dataset = 'ris'
baselines = ['ours2', 'context_only', 'crop_region', 'draw_region', 'miccai2023', 'region_in_text_sep']

for baseline in baselines:

    # define paths
    path_data = '/storage/homefs/st20f757/vqa/locvqallm/data/ris/processed'
    path_preds = f'/storage/homefs/st20f757/vqa/locvqallm/save/{dataset}/{version}_{baseline}/result'

    def clean_answer(answer):
        return answer[0].replace(' .', '')

    # load anns
    with open(jp(path_data, 'anns.json'), 'r') as f:
        anns = json.load(f)['test'] # contains q_id, and the answer is in the 'report' key, without dot

    # load gt answers
    with open(jp(path_preds, 'test_refs.json'), 'r') as f:
        gt_answers = json.load(f)
    # clean gt answers
    gt_answers = {k: clean_answer(v) for k, v in gt_answers.items()}

    # load preds
    with open(jp(path_preds, 'test_result.json'), 'r') as f:
        preds = json.load(f)
    # clean preds
    preds = {k: clean_answer(v) for k, v in preds.items()}

    all_objects = set([e['question_object'] for e in anns])

    # create empty dict with counters for each object
    objects = {k: 0 for k in all_objects}

    # iterate over anns, counting object occurrences and adding the corresponding prediction
    for e in anns:
        objects[e['question_object']] += 1
        e['pred'] = preds[e['q_id']]
        e['correct'] = int(e['pred'] == e['report'])

    # now for each object, count the number of correct predictions and divide by the total number of occurrences
    accs = {k: 0 for k in all_objects}
    for e in anns:
        accs[e['question_object']] += e['correct']

    # divide by the total number of occurrences
    for k in accs.keys():
        accs[k] /= objects[k]


    print('*'*20)
    print(f'Baseline: {baseline}')

    # print results with two decimal places
    for k, v in accs.items():
        print(f'{k}: {100*v:.2f}')


    # now compute overall accuracy by adding all correct predictions and dividing by the total number of predictions
    acc = sum([e['correct'] for e in anns]) / len(anns)
    print(f'Overall accuracy: {acc*100:.2f}')