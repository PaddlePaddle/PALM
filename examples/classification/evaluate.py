#  -*- coding: utf-8 -*-

import json
import numpy as np

def accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels) 
    return (preds == labels).mean()

def f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == '1') & (preds == '1'))
    tn = np.sum((labels == '0') & (preds == '0'))
    fp = np.sum((labels == '0') & (preds == '1'))
    fn = np.sum((labels == '1') & (preds == '0'))
    p = tp * 1.0 / (tp + fp) 
    r = tp * 1.0 / (tp + fn) * 1.0
    f1 = (2 * p * r) / (p + r + 1e-8)
    return f1
  
def recall(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    # recall=TP/(TP+FN)
    tp = np.sum((labels == '1') & (preds == '1'))
    fn = np.sum((labels == '1') & (preds == '0'))
    re = tp * 1.0 / (tp + fn)
    return re


def res_evaluate(res_dir="./outputs/predict/predictions.json", eval_phase='test'):
    if eval_phase == 'test':
        data_dir="./data/test.tsv"
    elif eval_phase == 'dev':
        data_dir="./data/dev.tsv"
    else:
        assert eval_phase in ['dev', 'test'], 'eval_phase should be dev or test'
    
    labels = []
    with open(data_dir, "r") as file:
        first_flag = True
        for line in file:
            line = line.split("\t")
            label = line[0]
            if label=='label':
                continue
            labels.append(str(label))
    file.close()

    preds = []
    with open(res_dir, "r") as file:
        for line in file.readlines():
            line = json.loads(line)
            pred = line['label']
            preds.append(str(pred))
    file.close()
    assert len(labels) == len(preds), "prediction result doesn't match to labels"
    print('data num: {}'.format(len(labels)))
    print("precision: {}, recall: {}, f1: {}".format(accuracy(preds, labels), recall(preds, labels), f1(preds, labels)))

res_evaluate()
