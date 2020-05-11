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
    num_classes = 130
    label_map = './data/atis/atis_slot/label_map.json'
    vocab_path = './pretrain/ERNIE-v2-en-base/vocab.txt'
    predict_file = './data/atis/atis_slot/test.tsv'
    save_path = './outputs/'
    pred_output = './outputs/predict-slot/'
    save_type = 'ckpt'
    random_seed = 0
    config = json.load(open('./pretrain/ERNIE-v2-en-base/ernie_config.json'))
    input_dim = config['hidden_size']

    # -----------------------  for prediction ----------------------- 

    # step 1-1: create readers for prediction
    print('prepare to predict...')
    predict_seq_label_reader = palm.reader.SequenceLabelReader(vocab_path, max_seqlen, label_map, seed=random_seed, phase='predict')
    # step 1-2: load the training data
    predict_seq_label_reader.load_data(predict_file, batch_size)
   
    # step 2: create a backbone of the model to extract text features
    pred_ernie = palm.backbone.ERNIE.from_config(config, phase='predict')
    
    # step 3: register the backbone in reader
    predict_seq_label_reader.register_with(pred_ernie)

    # step 4: create the task output head
    seq_label_pred_head = palm.head.SequenceLabel(num_classes, input_dim, phase='predict')
    
    # step 5-1: create a task trainer
    trainer_seq_label = palm.Trainer("slot")
    # step 5-2: build forward graph with backbone and task head
    trainer_seq_label.build_predict_forward(pred_ernie, seq_label_pred_head)
    
    # step 6: load checkpoint
    pred_model_path = './outputs/ckpt.step4641'
    trainer_seq_label.load_ckpt(pred_model_path)
    
    # step 7: fit prepared reader and data
    trainer_seq_label.fit_reader(predict_seq_label_reader, phase='predict')
   
    # step 8: predict
    print('predicting..')
    trainer_seq_label.predict(print_steps=print_steps, output_dir=pred_output)
