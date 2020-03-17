# coding=utf-8
import paddlepalm as palm
import json


if __name__ == '__main__':

    # configs
    max_seqlen = 256
    batch_size = 8
    vocab_path = './pretrain/ERNIE-v1-zh-base/vocab.txt'
    predict_file = './data/test.tsv'
    random_seed = 1
    config = json.load(open('./pretrain/ERNIE-v1-zh-base/ernie_config.json'))
    input_dim = config['hidden_size']
    num_classes = 2
    task_name = 'chnsenticorp'
    pred_output = './outputs/predict/'
    print_steps = 20
    pre_params = './pretrain/ERNIE-v1-zh-base/params'

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
    
    # step 5-1: create a task trainer
    trainer = palm.Trainer(task_name)
    # step 5-2: build forward graph with backbone and task head
    trainer.build_predict_forward(pred_ernie, cls_pred_head)
 
    # step 6: load checkpoint
    trainer.load_predict_model(pre_params)

    # step 7: fit prepared reader and data
    trainer.fit_reader(predict_cls_reader, phase='predict')

    # step 8: predict
    print('predicting..')
    trainer.predict(print_steps=print_steps, output_dir=pred_output)
