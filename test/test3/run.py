# coding=utf-8
import paddlepalm as palm
import json

if __name__ == '__main__':

    max_seqlen = 256
    batch_size = 8
    num_epochs = 10
    lr = 5e-5
    vocab_path = './pretrain/ernie-ch-uncased-base/vocab.txt'

    train_file = './data/chnsenticorp/train.tsv'
    predict_file = './data/chnsenticorp/test.tsv'
    random_seed = 1
    config = json.load(open('./pretrain/ernie-ch-uncased-base/ernie_config.json'))
    # ernie = palm.backbone.ERNIE(...)
    ernie = palm.backbone.ERNIE.from_config(config)

    # cls_reader2 = palm.reader.cls(train_file_topic, vocab_path, batch_size, max_seqlen)
    # cls_reader3 = palm.reader.cls(train_file_subj, vocab_path, batch_size, max_seqlen)
    # topic_trainer = palm.Trainer('topic_cls', cls_reader2, cls)
    # subj_trainer = palm.Trainer('subj_cls', cls_reader3, cls)

    # 创建该分类任务的reader，由诸多参数控制数据集读入格式、文件数量、预处理规则等
    cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, seed=random_seed)
    predict_cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, seed=random_seed, phase='predict')
    # 不同的backbone会对任务reader有不同的特征要求，例如对于分类任务，基本的输入feature为token_ids和label_ids，但是对于BERT，还要求从输入中额外提取position、segment、input_mask等特征，因此经过register后，reader会自动补充backbone所要求的字段
    cls_reader.register_with(ernie)

    print("preparing data...")
    print(cls_reader.num_examples)
    cls_reader.load_data(train_file, batch_size, num_epochs=num_epochs)
    print(cls_reader.num_examples)
    print('done!')
    input_dim = config['hidden_size']
    num_classes = 2
    dropout_prob = 0.1
    random_seed = 1
    # 创建任务头（task head），如分类、匹配、机器阅读理解等。每个任务头有跟该任务相关的必选/可选参数。注意，任务头与reader是解耦合的，只要任务头依赖的数据集侧的字段能被reader提供，那么就是合法的
    cls_head = palm.head.Classify(num_classes, input_dim, dropout_prob)

    # 根据reader和任务头来创建一个训练器trainer，trainer代表了一个训练任务，内部维护着训练进程、和任务的关键信息，并完成合法性校验，该任务的模型保存、载入等相关规则控制
    trainer = palm.Trainer('senti_cls')


    loss_var = trainer.build_forward(ernie, cls_head)


    n_steps = cls_reader.num_examples * num_epochs // batch_size
    # warmup_steps = int(0.1 * n_steps)
    # print(warmup_steps)
    """
    # sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)
    sched = None

    adam = palm.optimizer.Adam(loss_var, lr, sched)

    trainer.build_backward(optimizer=adam, weight_decay=0.01)

    trainer.random_init_params()
    trainer.load_pretrain('pretrain/ernie-ch-uncased-base/params')

    trainer.fit_reader(cls_reader)
    trainer.train(print_steps=1, save_steps=n_steps-24, save_path='outputs', save_type='ckpt')
    """
    model_path = './outputs/ckpt.step'+str(n_steps-24)
    print('prepare to predict...')
    pred_ernie = palm.backbone.ERNIE.from_config(config, phase='predict')
    predict_cls_reader.register_with(pred_ernie)
    predict_cls_reader.load_data(predict_file, 8)
    
    cls_pred_head = palm.head.Classify(num_classes, input_dim, phase='predict')
    trainer.build_predict_forward(pred_ernie, cls_pred_head)
    pred_ckpt = trainer.load_ckpt(model_path, phase='predict')
    trainer.fit_reader(predict_cls_reader, phase='predict')
    
    print(predict_cls_reader.num_examples)
    
    print('predicting..')
    trainer.predict(print_steps=20, output_dir="outputs/test/")
