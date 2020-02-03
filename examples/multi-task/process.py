# -*- coding: UTF-8 -*-
import json
import os
import io

abs_path = os.path.abspath(__file__)
dst_dir = os.path.join(os.path.dirname(abs_path), "data/mlm/")
dst_dir2 = os.path.join(os.path.dirname(abs_path), "data/match/")
if not os.path.exists(dst_dir) or not os.path.isdir(dst_dir):
    os.makedirs(dst_dir)
if not os.path.exists(dst_dir2) or not os.path.isdir(dst_dir2):
    os.makedirs(dst_dir2)
os.mknod("./data/mlm/train.tsv")
os.mknod("./data/match/train.tsv")

with io.open("./data/mrc/train.json", "r", encoding='utf-8') as file:
    data = json.load(file)["data"] 
    i = 0
    with open("./data/mlm/train.tsv","w") as f:
        f.write("text_a\n")
        with open("./data/match/train.tsv","w") as f2:
            f2.write("text_a\ttext_b\tlabel\n") 
            for dd in data:
                for d in dd["paragraphs"]: 
                    text_a_mlm = d["context"]
                    l = text_a_mlm+"\n"
                    f.write(l.encode("utf-8"))
                for qa in d["qas"]:
                    text_a = qa["question"]
                    answer = qa["answers"][0]
                    text_b = answer["text"]
                    start_pos = answer["answer_start"]
                    text_b_neg = text_a_mlm[0:start_pos]
                    if len(text_b_neg) > 512:
                        text_b_neg = text_b_neg[-512:-1]
                    l1 = text_a+"\t"+text_b+"\t1\n" 
                    l2 = text_a+"\t"+text_b_neg+"\t0\n"
                    if i < 14246:
                        f2.write(l1.encode("utf-8"))
                        f2.write(l2.encode("utf-8"))
                        i +=2

        f2.close()
    f.close()
file.close()
