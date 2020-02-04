import os
import json

label_new = "data/atis/atis_slot/label_map.json"
label_old = "data/atis/atis_slot/map_tag_slot_id.txt"
train_old = "data/atis/atis_slot/train.txt"
train_new = "data/atis/atis_slot/train.tsv"
dev_old = "data/atis/atis_slot/dev.txt"
dev_new = "data/atis/atis_slot/dev.tsv"
test_old = "data/atis/atis_slot/test.txt"
test_new = "data/atis/atis_slot/test.tsv"


intent_test =  "data/atis/atis_intent/test.tsv"
os.rename("data/atis/atis_intent/test.txt", intent_test)
intent_train =  "data/atis/atis_intent/train.tsv"
os.rename("data/atis/atis_intent/train.txt", intent_train)
intent_dev = "data/atis/atis_intent/dev.tsv"
os.rename("data/atis/atis_intent/dev.txt", intent_dev)

with open(intent_dev, 'r+') as f: 
    content = f.read()  
    f.seek(0, 0)
    f.write("label\ttext_a\n"+content)
f.close()

with open(intent_test, 'r+') as f: 
    content = f.read()  
    f.seek(0, 0)
    f.write("label\ttext_a\n"+content)
f.close()

with open(intent_train, 'r+') as f: 
    content = f.read()  
    f.seek(0, 0)
    f.write("label\ttext_a\n"+content)
f.close()

os.mknod(label_new)
os.mknod(train_new)
os.mknod(dev_new)
os.mknod(test_new)


tag = []
id = []
map = {}
with open(label_old, "r") as f:
    with open(label_new, "w") as f2:
        for line in f.readlines():
            line = line.split('\t')
            tag.append(line[0])
            id.append(int(line[1][:-1]))
            map[line[1][:-1]] = line[0]

        re = {tag[i]:id[i] for i in range(len(tag))}
        re = json.dumps(re)
        f2.write(re)
    f2.close()
f.close()


with open(train_old, "r") as f:
    with open(train_new, "w") as f2:
        f2.write("text_a\tlabel\n")
        for line in f.readlines():
            line = line.split('\t')
            text = line[0].split(' ')
            label = line[1].split(' ')
            for t in text:
                f2.write(t)
                f2.write('\2')
            f2.write('\t')
            for t in label:
                if t.endswith('\n'):
                    t = t[:-1] 
                f2.write(map[t])
                f2.write('\2')
            f2.write('\n')
    f2.close()
f.close()

with open(test_old, "r") as f:
    with open(test_new, "w") as f2:
        f2.write("text_a\tlabel\n")
        for line in f.readlines():
            line = line.split('\t')
            text = line[0].split(' ')
            label = line[1].split(' ')
            for t in text:
                f2.write(t)
                f2.write('\2')
            f2.write('\t')
            for t in label:
                if t.endswith('\n'):
                    t = t[:-1] 
                f2.write(map[t])
                f2.write('\2')
            f2.write('\n')
    f2.close()
f.close()

with open(dev_old, "r") as f:
    with open(dev_new, "w") as f2:
        f2.write("text_a\tlabel\n")
        for line in f.readlines():
            line = line.split('\t')
            text = line[0].split(' ')
            label = line[1].split(' ')
            for t in text:
                f2.write(t)
                f2.write('\2')
            f2.write('\t')
            for t in label:
                if t.endswith('\n'):
                    t = t[:-1] 
                f2.write(map[t])
                f2.write('\2')
            f2.write('\n')
    f2.close()
f.close()

os.remove(label_old)
os.remove(train_old)
os.remove(test_old)
os.remove(dev_old)