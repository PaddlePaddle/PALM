# coding=utf-8
import paddlepalm as palm
import json
import numpy as np


if __name__ == '__main__':

    # configs
    max_seqlen = 128
    batch_size = 128
    num_epochs = 20
    print_steps = 5
    lr = 2e-5
    num_classes = 130
    weight_decay = 0.01
    num_classes_intent = 26
    dropout_prob = 0.1
    random_seed = 0
    label_map = './data/atis/atis_slot/label_map.json'
    vocab_path = './pretrain/ERNIE-v2-en-base/vocab.txt'

    train_slot = './data/atis/atis_slot/train.tsv'
    train_intent = './data/atis/atis_intent/train.tsv'

    config = json.load(open('./pretrain/ERNIE-v2-en-base/ernie_config.json'))
    input_dim = config['hidden_size']

    # -----------------------  for training ----------------------- 

    # step 1-1: create readers 
    slot_reader = palm.reader.SequenceLabelReader(vocab_path, max_seqlen, label_map, seed=random_seed, phase='predict')
    intent_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, seed=random_seed, phase='predict')

    # step 1-2: load train data
    slot_reader.load_data(train_slot, file_format='tsv', num_epochs=None, batch_size=batch_size)
    intent_reader.load_data(train_intent, batch_size=batch_size, num_epochs=None)

    # step 2: create a backbone of the model to extract text features
    ernie = palm.backbone.ERNIE.from_config(config, phase='predict')

    # step 3: register readers with ernie backbone
    slot_reader.register_with(ernie)
    intent_reader.register_with(ernie)

    # step 4: create task output heads
    slot_head = palm.head.SequenceLabel(num_classes, input_dim, dropout_prob, phase='predict')
    intent_head = palm.head.Classify(num_classes_intent, input_dim, dropout_prob, phase='predict')
   
    # step 5-1: create task trainers and multiHeadTrainer
    trainer_slot = palm.Trainer("slot", mix_ratio=1.0)
    trainer_intent = palm.Trainer("intent", mix_ratio=1.0)
    trainer = palm.MultiHeadTrainer([trainer_slot, trainer_intent])
    # # step 5-2: build forward graph with backbone and task head
    vars = trainer_intent.build_predict_forward(ernie, intent_head)
    vars = trainer_slot.build_predict_forward(ernie, slot_head)
    loss_var = trainer.build_predict_forward()

    # load checkpoint
    trainer.load_ckpt('outputs/ckpt.step300')

    # merge inference readers
    joint_iterator = trainer.merge_inference_readers([slot_reader, intent_reader])

    # for test
    # batch = next(joint_iterator('slot'))
    # results = trainer.predict_one_batch('slot', batch)
    # batch = next(joint_iterator('intent'))
    # results = trainer.predict_one_batch('intent', batch)

    # predict slot filling
    print('processing slot filling examples...')
    print('num examples: '+str(slot_reader.num_examples))
    cnt = 0
    for batch in joint_iterator('slot'):
        cnt += len(trainer.predict_one_batch('slot', batch)['logits'])
        if cnt % 1000 <= 128:
            print(str(cnt)+'th example processed.')
    print(str(cnt)+'th example processed.')

    # predict intent recognition
    print('processing intent recognition examples...')
    print('num examples: '+str(intent_reader.num_examples))
    cnt = 0
    for batch in joint_iterator('intent'):
        cnt += len(trainer.predict_one_batch('intent', batch)['logits'])
        if cnt % 1000 <= 128:
            print(str(cnt)+'th example processed.')
    print(str(cnt)+'th example processed.')

