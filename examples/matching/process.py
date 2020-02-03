#  -*- coding: utf-8 -*-

import sys
import os

if len(sys.argv) != 4:
    exit(0)

data_dir = sys.argv[1]
if not os.path.exists(data_dir):
    print("%s not exists" % data_dir)
    exit(0)

train_dir = sys.argv[2]
train_file = open(train_dir, "w")
train_file.write("text_a\ttext_b\tlabel\n")

test_dir = sys.argv[3]
test_file = open(test_dir, "w")
test_file.write("text_a\ttext_b\tlabel\n")
with open(data_dir, "r") as file:
    before = ""
    cnt = 0
    for line in file:
        line = line.strip("\n")
        line_t = line.split("\t")
        flag = 0
        if len(line_t) < 6:
            if flag: 
                flag = 0
                out_line = "{}{}\n".format(out_line, line)
            else:
                flag = 1
                outline = "{}".format(line)
            continue
        else:
            out_line = "{}\t{}\t{}\n".format(line_t[3], line_t[4], line_t[5])
        cnt += 1

        if 2 <= cnt <= 4301:
            test_file.write(out_line)
        if 4301 <= cnt <= 104301:
            train_file.write(out_line)

train_file.close()
test_file.close()
