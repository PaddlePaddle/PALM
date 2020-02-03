# coding=utf-8
import paddlepalm as palm
import json

if __name__ == '__main__':

    max_seqlen = 512
    batch_size = 4
    num_epochs = 2
    lr = 1e-3
    vocab_path = './pretrain/ernie/vocab.txt'

    train_file = './data/cls4mrqa/train.tsv'
    predict_file = './data/cls4mrqa/dev.tsv'

    config = json.load(open('./pretrain/ernie/ernie_config.json'))
    # ernie = palm.backbone.ERNIE(...)
    ernie = palm.backbone.ERNIE.from_config(config)

    # cls_reader2 = palm.reader.cls(train_file_topic, vocab_path, batch_size, max_seqlen)
    # cls_reader3 = palm.reader.cls(train_file_subj, vocab_path, batch_size, max_seqlen)
    # topic_trainer = palm.Trainer('topic_cls', cls_reader2, cls)
    # subj_trainer = palm.Trainer('subj_cls', cls_reader3, cls)

    # 创建该分类任务的reader，由诸多参数控制数据集读入格式、文件数量、预处理规则等
    cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen)
<<<<<<< HEAD:test/test2/run.py
    cls_reader2 = palm.reader.ClassifyReader(vocab_path, max_seqlen)
=======
    predict_cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen, phase='predict')
>>>>>>> remotes/upstream/r0.3-api:test/test3/run.py
    print(cls_reader.outputs_attr)
    print(predict_cls_reader.outputs_attr)
    # 不同的backbone会对任务reader有不同的特征要求，例如对于分类任务，基本的输入feature为token_ids和label_ids，但是对于BERT，还要求从输入中额外提取position、segment、input_mask等特征，因此经过register后，reader会自动补充backbone所要求的字段
    cls_reader.register_with(ernie)
    cls_reader2.register_with(ernie)
    print(cls_reader.outputs_attr)
<<<<<<< HEAD:test/test2/run.py

    print("preparing data...")
    print(cls_reader.num_examples)
    cls_reader.load_data(train_file, batch_size)
    cls_reader2.load_data(train_file, batch_size)
=======
    print(predict_cls_reader.outputs_attr)

    print("preparing data...")
    print(cls_reader.num_examples)
    cls_reader.load_data(train_file, batch_size, num_epochs=num_epochs)
>>>>>>> remotes/upstream/r0.3-api:test/test3/run.py
    print(cls_reader.num_examples)
    print('done!')

    # 创建任务头（task head），如分类、匹配、机器阅读理解等。每个任务头有跟该任务相关的必选/可选参数。注意，任务头与reader是解耦合的，只要任务头依赖的数据集侧的字段能被reader提供，那么就是合法的
    cls_head = palm.head.Classify(4, 1024, 0.1)
<<<<<<< HEAD:test/test2/run.py
    cls_head2 = palm.head.Classify(4, 1024, 0.1)

    # 根据reader和任务头来创建一个训练器trainer，trainer代表了一个训练任务，内部维护着训练进程、和任务的关键信息，并完成合法性校验，该任务的模型保存、载入等相关规则控制
    trainer = palm.Trainer('cls')
    trainer2 = palm.Trainer('senti_cls')
    mh_trainer = palm.MultiHeadTrainer([trainer, trainer2])
=======

    # 根据reader和任务头来创建一个训练器trainer，trainer代表了一个训练任务，内部维护着训练进程、和任务的关键信息，并完成合法性校验，该任务的模型保存、载入等相关规则控制
    trainer = palm.Trainer('senti_cls')
>>>>>>> remotes/upstream/r0.3-api:test/test3/run.py

    # match4mrqa.reuse_head_with(mrc4mrqa)

    # data_vars = cls_reader.build()
    # output_vars = ernie.build(data_vars)
    # cls_head.build({'backbone': output_vars, 'reader': data_vars})

<<<<<<< HEAD:test/test2/run.py
    loss_var = mh_trainer.build_forward(ernie, [cls_head, cls_head2])

    n_steps = cls_reader.num_examples * num_epochs // batch_size
    warmup_steps = int(0.1 * n_steps)
    print(warmup_steps)
    sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)
=======
    loss_var = trainer.build_forward(ernie, cls_head)

    # controller.build_forward()
    # Error! a head/backbone can be only build once! Try NOT to call build_forward method for any Trainer!

    # n_steps = cls_reader.num_examples * num_epochs // batch_size
    # warmup_steps = int(0.1 * n_steps)
    # print(warmup_steps)
    # sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)
    sched = None
>>>>>>> remotes/upstream/r0.3-api:test/test3/run.py

    adam = palm.optimizer.Adam(loss_var, lr, sched)

    mh_trainer.build_backward(optimizer=adam, weight_decay=0.001)
    
    # mh_trainer.random_init_params()
    mh_trainer.load_pretrain('pretrain/ernie/params')

    # trainer.train(iterator_fn, print_steps=1, save_steps=5, save_path='outputs', save_type='ckpt,predict')
<<<<<<< HEAD:test/test2/run.py
    mh_trainer.fit_readers_with_mixratio([cls_reader, cls_reader2], 'cls', 2)
    mh_trainer.train(print_steps=1)
    # trainer.save()

=======
    trainer.fit_reader(cls_reader)
    trainer.train(print_steps=1)
    # trainer.save()

    print('prepare to predict...')
    pred_ernie = palm.backbone.ERNIE.from_config(config, phase='pred')
    cls_pred_head = palm.head.Classify(4, 1024, phase='pred')
    trainer.build_predict_forward(pred_ernie, cls_pred_head)

    predict_cls_reader.load_data(predict_file, 8)
    print(predict_cls_reader.num_examples)
    predict_cls_reader.register_with(pred_ernie)
    trainer.fit_reader(predict_cls_reader, phase='predict')
    print('predicting..')
    trainer.predict(print_steps=20)








    # controller = palm.Controller([mrqa, match4mrqa, mlm4mrqa])

    # loss = controller.build_forward(bb, mask_task=[])

    # n_steps = controller.estimate_train_steps(basetask=mrqa, num_epochs=2, batch_size=8, dev_count=4)
    # adam = palm.optimizer.Adam(loss)
    # sched = palm.schedualer.LinearWarmup(learning_rate, max_train_steps=n_steps, warmup_steps=0.1*n_steps)
    # 
    # controller.build_backward(optimizer=adam, schedualer=sched, weight_decay=0.001, use_ema=True, ema_decay=0.999)

    # controller.random_init_params()
    # controller.load_pretrain('../../pretrain_model/ernie/params')
    # controller.train()





    # controller = palm.Controller(config='config.yaml', task_dir='tasks', for_train=False)
    # controller.pred('mrqa', inference_model_dir='output_model/secondrun/mrqa/infer_model')


>>>>>>> remotes/upstream/r0.3-api:test/test3/run.py
