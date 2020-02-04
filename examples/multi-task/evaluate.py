#  -*- coding: utf-8 -*-

import json


def load_label_map(map_dir="./data/atis/atis_slot/label_map.json"):
    """
    :param map_dir: dict indictuing chunk type
    :return:
    """
    return json.load(open(map_dir, "r"))


def cal_chunk(total_res, total_label):
    assert len(total_label) == len(total_res), 'prediction result doesn\'t match to labels'
    num_labels = 0
    num_corr = 0
    num_infers = 0
    for res, label in zip(total_res, total_label):
        assert len(res) == len(label), "prediction result doesn\'t match to labels"
        num_labels += sum([0 if i == 6 else 1 for i in label])
        num_corr += sum([1 if label[i] == res[i] and label[i] != 6 else 0 for i in range(len(label))])
        num_infers += sum([0 if i == 6 else 1 for i in res])

    precision = num_corr * 1.0 / num_infers if num_infers > 0 else 0.0
    recall = num_corr * 1.0 / num_labels if num_labels > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1


def res_evaluate(res_dir="./outputs/predict/predictions.json", data_dir="./data/atis/atis_slot/test.tsv"):
    label_map = load_label_map()

    total_label = []
    with open(data_dir, "r") as file:
        first_flag = True
        for line in file:
            if first_flag:
                first_flag = False
                continue
            line = line.strip("\n")
            if len(line) == 0:
                continue
            line = line.split("\t")
            if len(line) < 2:
                continue
            labels = line[1][:-1].split("\x02")
            total_label.append(labels)
    total_label = [[label_map[j] for j in i] for i in total_label]

    total_res = []
    with open(res_dir, "r") as file:
        cnt = 0
        for line in file:
            line = line.strip("\n")
            if len(line) == 0:
                continue
            try:
                res_arr = json.loads(line)

                if len(total_label[cnt]) < len(res_arr):
                    total_res.append(res_arr[1: 1 + len(total_label[cnt])])
                elif len(total_label[cnt]) == len(res_arr):
                    total_res.append(res_arr)
                else:
                    total_res.append(res_arr)
                    total_label[cnt] = total_label[cnt][: len(res_arr)]
            except:
                print("json format error: {}".format(cnt))
                print(line)

            cnt += 1

    precision, recall, f1 = cal_chunk(total_res, total_label)
    print("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))

res_evaluate()
