# coding: utf-8
f='mrqa-combined.train.raw.json'
import json
a=json.load(open(f))
a=a['data']
writer = open('train.json','w')
    
for s in a:
    p = s['paragraphs']
    assert len(p) == 1
    p = p[0]
    q = {}
    q['context'] = p['context']
    q['qa_list'] = p['qas']
    writer.write(json.dumps(q)+'\n')
    
writer.close()
    
