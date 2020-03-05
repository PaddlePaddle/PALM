# coding=utf-8
import paddlepalm as palm
import json
from paddlepalm.distribute import gpu_dev_count


if __name__ == '__main__':

    # configs
    max_seqlen = 256
    batch_size = 16
    num_epochs = 6 
    print_steps = 5
    num_classes = 26
    vocab_path = './pretrain/ERNIE-v2-en-base/vocab.txt'
    predict_file = './data/atis/atis_intent/test.tsv'
    save_path = './outputs/'
    pred_output = './outputs/predict-intent/'
    save_type = 'ckpt'
    random_seed = 0
    config = json.load(open('./pretrain/ERNIE-v2-en-base/ernie_config.json'))
    input_dim = config['hidden_size']

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
    trainer = palm.Trainer("intent")
    # step 5-2: build forward graph with backbone and task head
    trainer.build_predict_forward(pred_ernie, cls_pred_head)
 
    # step 6: load checkpoint
    pred_model_path = './outputs/ckpt.step4641'
    trainer.load_ckpt(pred_model_path)

    # step 7: fit prepared reader and data
    trainer.fit_reader(predict_cls_reader, phase='predict')

    # step 8: predict
    print('predicting..')
    trainer.predict(print_steps=print_steps, output_dir=pred_output)
