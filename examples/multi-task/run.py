# coding=utf-8
import paddlepalm as palm
import json
from paddlepalm.distribute import gpu_dev_count


if __name__ == '__main__':

    # configs
    max_seqlen = 512
    batch_size = 8   
    num_epochs = 8
    lr = 3e-5
    doc_stride = 128
    max_query_len = 64
    max_ans_len = 128
    weight_decay = 0.01
    print_steps = 20
    num_classes = 2
    random_seed = 1
    dropout_prob = 0.1
    vocab_path = './pretrain/ernie-zh-base/vocab.txt'
    do_lower_case = True

    train_file = './data/mrc/train.json'
    train_file_mlm = './data/mlm/train.tsv'
    train_file_match = './data/match/train.tsv'
    predict_file = './data/mrc/dev.json'
    save_path = './outputs/'
    pred_output = './outputs/predict/'
    save_type = 'ckpt'
    task_name = 'cmrc2018'
    pre_params = './pretrain/ernie-zh-base/params'
    config = json.load(open('./pretrain/ernie-zh-base/ernie_config.json'))
    input_dim = config['hidden_size']
    vocab_size = config['vocab_size']
    hidden_act = config['hidden_act']

    # -----------------------  for training ----------------------- 

    # step 1-1: create readers for training
    mrc_reader = palm.reader.MRCReader(vocab_path, max_seqlen, max_query_len, doc_stride, do_lower_case=do_lower_case)
    match_reader = palm.reader.MatchReader(vocab_path, max_seqlen, seed=random_seed)
    # mlm_reader = palm.reader.MaskLMReader(vocab_path, max_seqlen, seed=random_seed)
    # step 1-2: load the training data
    mrc_reader.load_data(train_file, file_format='json', num_epochs=None, batch_size=batch_size)
    match_reader.load_data(train_file_match, file_format='tsv', num_epochs=None, batch_size=batch_size)
    # mlm_reader.load_data(train_file_mlm, file_format='tsv', num_epochs=num_epochs, batch_size=batch_size)

    # step 2: create a backbone of the model to extract text features
    ernie = palm.backbone.ERNIE.from_config(config)

    # step 3: register the backbone in readers
    mrc_reader.register_with(ernie)
    match_reader.register_with(ernie)
    # mlm_reader.register_with(ernie)

    # step 4: create task output heads
    mrc_head = palm.head.MRC(max_query_len, config['hidden_size'], do_lower_case=do_lower_case, max_ans_len=max_ans_len)
    match_head = palm.head.Match(num_classes, input_dim, dropout_prob)
    mlm_head = palm.head.MaskLM(input_dim, hidden_act, dropout_prob)

    # step 5-1: create a task trainer
    trainer_mrc = palm.Trainer(task_name, mix_ratio=1.0)
    # trainer_mlm = palm.Trainer("mlm", mix_ratio=0.5)
    trainer_match = palm.Trainer("match", mix_ratio=0.5)
    trainer = palm.MultiHeadTrainer([trainer_mrc, trainer_match])
    # step 5-2: build forward graph with backbone and task head
    loss_var = trainer.build_forward(ernie, [mrc_head, match_head])
    
    # step 6-1*: use warmup
    n_steps = mrc_reader.num_examples * num_epochs // batch_size
    warmup_steps = int(0.1 * n_steps)
    sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)
    # step 6-2: create a optimizer
    adam = palm.optimizer.Adam(loss_var, lr, sched)
    # step 6-3: build backward
    trainer.build_backward(optimizer=adam, weight_decay=weight_decay)

    # step 7: fit prepared reader and data
    trainer.fit_readers_with_mixratio([mrc_reader, match_reader], task_name, num_epochs)
 
    # step 8-1*: load pretrained parameters
    trainer.load_pretrain(pre_params)
    # step 8-2*: set saver to save model
    # save_steps = n_steps-8
    save_steps = 1520
    trainer.set_saver(save_path=save_path, save_steps=save_steps, save_type=save_type)
    # step 8-3: start training
    trainer.train(print_steps=print_steps)
   
    # -----------------------  for prediction ----------------------- 

    # step 1-1: create readers for prediction
    predict_mrc_reader = palm.reader.MRCReader(vocab_path, max_seqlen, max_query_len, doc_stride, do_lower_case=do_lower_case, phase='predict')
    # step 1-2: load the training data
    predict_mrc_reader.load_data(predict_file, batch_size)

    # step 2: create a backbone of the model to extract text features
    pred_ernie = palm.backbone.ERNIE.from_config(config, phase='predict')

    # step 3: register the backbone in reader
    predict_mrc_reader.register_with(pred_ernie)

    # step 4: create the task output head
    mrc_pred_head = palm.head.MRC(max_query_len, config['hidden_size'], do_lower_case=do_lower_case, max_ans_len=max_ans_len, phase='predict')
    
    # step 5: build forward graph with backbone and task head
    trainer.build_predict_forward(pred_ernie, mrc_pred_head)

    # step 6: load pretrained model
    pred_model_path =  './outputs/ckpt.step'+str(12160)
    pred_ckpt = trainer.load_ckpt(pred_model_path)
    
    # step 7: fit prepared reader and data
    trainer.fit_reader(predict_mrc_reader, phase='predict')

    # step 8: predict
    print('predicting..')
    trainer.predict(print_steps=print_steps, output_dir="outputs/")
