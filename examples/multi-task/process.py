# -*- coding: UTF-8 -*-
import json
import os
import io

abs_path = os.path.abspath(__file__)
dst_dir = os.path.join(os.path.dirname(abs_path), "data/match/")
if not os.path.exists(dst_dir) or not os.path.isdir(dst_dir):
    os.makedirs(dst_dir)
os.mknod("./data/match/train.tsv")

with io.open("./data/mrc/train.json", "r", encoding='utf-8') as f:
    data = json.load(f)["data"] 
    i = 0
    with open("./data/match/train.tsv","w") as f2:
        f2.write("text_a\ttext_b\tlabel\n") 
        for dd in data:
            for d in dd["paragraphs"]: 
                context = d["context"]
                for qa in d["qas"]:
                    text_a = qa["question"]
                    answer = qa["answers"][0]
                    text_b = answer["text"]
                    start_pos = answer["answer_start"]
                    text_b_neg = context[0:start_pos]
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
