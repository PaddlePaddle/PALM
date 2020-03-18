# coding=utf-8
import paddlepalm as palm
import json


if __name__ == '__main__':

    # configs
    max_seqlen = 256
    batch_size = 8
    num_epochs = 10
    lr = 5e-5
    weight_decay = 0.01
    vocab_path = './pretrain/ERNIE-v1-zh-base/vocab.txt'

    train_file = './data/train.tsv'
    predict_file = './data/test.tsv'
    config = json.load(open('./pretrain/ERNIE-v1-zh-base/ernie_config.json'))
    input_dim = config['hidden_size']
    num_classes = 2
    dropout_prob = 0.1
    random_seed = 1
    task_name = 'chnsenticorp'
    save_path = './outputs/'
    pred_output = './outputs/predict/'
    save_type = 'ckpt'
    print_steps = 20
    pre_params = './pretrain/ERNIE-v1-zh-base/params'

    # -----------------------  for training ----------------------- 

    # step 1-1: create readers for training
    cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, seed=random_seed)
    # step 1-2: load the training data
    cls_reader.load_data(train_file, batch_size, num_epochs=num_epochs)

    # step 2: create a backbone of the model to extract text features
    ernie = palm.backbone.ERNIE.from_config(config)

    # step 3: register the backbone in reader
    cls_reader.register_with(ernie)

    # step 4: create the task output head
    cls_head = palm.head.Classify(num_classes, input_dim, dropout_prob)

    # step 5-1: create a task trainer
    trainer = palm.Trainer(task_name)
    # step 5-2: build forward graph with backbone and task head
    loss_var = trainer.build_forward(ernie, cls_head)

    # step 6-1*: use warmup
    n_steps = cls_reader.num_examples * num_epochs // batch_size
    warmup_steps = int(0.1 * n_steps)
    sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)
    # step 6-2: create a optimizer
    adam = palm.optimizer.Adam(loss_var, lr, sched)
    # step 6-3: build backward
    trainer.build_backward(optimizer=adam, weight_decay=weight_decay)
  
    # step 7: fit prepared reader and data
    trainer.fit_reader(cls_reader)
    
    # step 8-1*: load pretrained parameters
    trainer.load_pretrain(pre_params)
    # step 8-2*: set saver to save model
    # save_steps = n_steps 
    save_steps = 2396
    trainer.set_saver(save_steps=save_steps, save_path=save_path, save_type=save_type)
    # step 8-3: start training
    trainer.train(print_steps=print_steps)
   
    # -----------------------  for prediction ----------------------- 

    # step 1-1: create readers for prediction
    print('prepare to predict...')
    predict_cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, seed=random_seed, phase='predict')
    # step 1-2: load the training data
    predict_cls_reader.load_data(predict_file, batch_size)
    
    # step 2: create a backbone of the model to extract text features
    pred_ernie = palm.backbone.ERNIE.from_config(config, phase='predict')

    # step 3: register the backbone in reader
    predict_cls_reader.register_with(pred_ernie)
    
    # step 4: create the task output head
    cls_pred_head = palm.head.Classify(num_classes, input_dim, phase='predict')
    
    # step 5: build forward graph with backbone and task head
    trainer.build_predict_forward(pred_ernie, cls_pred_head)
 
    # step 6: load checkpoint
    # model_path = './outputs/ckpt.step'+str(save_steps)
    model_path = './outputs/ckpt.step'+str(11980)
    trainer.load_ckpt(model_path)

    # step 7: fit prepared reader and data
    trainer.fit_reader(predict_cls_reader, phase='predict')

    # step 8: predict
    print('predicting..')
    trainer.predict(print_steps=print_steps, output_dir=pred_output)
