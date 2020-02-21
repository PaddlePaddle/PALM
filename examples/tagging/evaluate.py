#  -*- coding: utf-8 -*-

import json


def load_label_map(map_dir="./data/label_map.json"):
    """
    :param map_dir: dict indictuing chunk type
    :return:
    """
    return json.load(open(map_dir, "r"))


def cal_chunk(pred_label, refer_label):
    tp = dict()
    fn = dict()
    fp = dict()
    for i in range(len(refer_label)):
        if refer_label[i] == pred_label[i]:
            if refer_label[i] not in tp:
                tp[refer_label[i]] = 0
            tp[refer_label[i]] += 1
        else:
            if pred_label[i] not in fp:
                fp[pred_label[i]] = 0
            fp[pred_label[i]] += 1
            if refer_label[i] not in fn:
                fn[refer_label[i]] = 0
            fn[refer_label[i]] += 1

    tp_total = sum(tp.values())
    fn_total = sum(fn.values())
    fp_total = sum(fp.values())
    p_total = float(tp_total) / (tp_total + fp_total)
    r_total = float(tp_total) / (tp_total + fn_total)
    f_micro = 2 * p_total * r_total / (p_total + r_total)

    return f_micro


def res_evaluate(res_dir="./outputs/predict/predictions.json", data_dir="./data/test.tsv"):
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
            labels = line[1].split("\x02")
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

    total_res_equal = []
    total_label_equal = []
    assert len(total_label) == len(total_res), "prediction result doesn't match to labels"
    for i in range(len(total_label)):
        num = len(total_label[i])
        total_label_equal.extend(total_label[i])
        total_res[i] = total_res[i][:num]
        total_res_equal.extend(total_res[i])

    f1 = cal_chunk(total_res_equal, total_label_equal)
    print('data num: {}'.format(len(total_label)))
    print("f1: {:.4f}".format(f1))

res_evaluate()
