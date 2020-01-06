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

    config = json.load(open('./pretrain/ernie/ernie_config.json'))
    # ernie = palm.backbone.ERNIE(...)
    ernie = palm.backbone.ERNIE.from_config(config)
    # pred_ernie = palm.backbone.ERNIE.from_config(config, phase='pred')

    # cls_reader2 = palm.reader.cls(train_file_topic, vocab_path, batch_size, max_seqlen)
    # cls_reader3 = palm.reader.cls(train_file_subj, vocab_path, batch_size, max_seqlen)
    # topic_trainer = palm.Trainer('topic_cls', cls_reader2, cls)
    # subj_trainer = palm.Trainer('subj_cls', cls_reader3, cls)

    # 创建该分类任务的reader，由诸多参数控制数据集读入格式、文件数量、预处理规则等
    cls_reader = palm.reader.ClassifyReader(vocab_path, max_seqlen)
    print(cls_reader.outputs_attr)
    # 不同的backbone会对任务reader有不同的特征要求，例如对于分类任务，基本的输入feature为token_ids和label_ids，但是对于BERT，还要求从输入中额外提取position、segment、input_mask等特征，因此经过register后，reader会自动补充backbone所要求的字段
    cls_reader.register_with(ernie)
    print(cls_reader.outputs_attr)
    # 创建任务头（task head），如分类、匹配、机器阅读理解等。每个任务头有跟该任务相关的必选/可选参数。注意，任务头与reader是解耦合的，只要任务头依赖的数据集侧的字段能被reader提供，那么就是合法的
    cls_head = palm.head.Classify(4, 1024, 0.1)
    # cls_pred_head = palm.head.Classify(4, 1024, 0.1, phase='pred')

    # 根据reader和任务头来创建一个训练器trainer，trainer代表了一个训练任务，内部维护着训练进程、和任务的关键信息，并完成合法性校验，该任务的模型保存、载入等相关规则控制
    trainer = palm.Trainer('senti_cls', cls_reader, cls_head)

    # match4mrqa.reuse_head_with(mrc4mrqa)

    # data_vars = cls_reader.build()
    # output_vars = ernie.build(data_vars)
    # cls_head.build({'backbone': output_vars, 'reader': data_vars})

    loss_var = trainer.build_forward(ernie)

    # controller.build_forward()
    # Error! a head/backbone can be only build once! Try NOT to call build_forward method for any Trainer!

    print(trainer.num_examples)
    iterator_fn = trainer.load_data(train_file, 'csv', num_epochs=num_epochs, batch_size=batch_size)
    print(trainer.num_examples)

    n_steps = trainer.num_examples * num_epochs // batch_size
    warmup_steps = int(0.1 * n_steps)
    print(warmup_steps)
    sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)

    adam = palm.optimizer.Adam(loss_var, lr, sched)

    trainer.build_backward(optimizer=adam, weight_decay=0.001)

    trainer.random_init_params()
    trainer.load_pretrain('pretrain/ernie/params')

    # print(trainer.train_one_step(next(iterator_fn())))
    # trainer.train_one_epoch()
    trainer.train(iterator_fn, print_steps=1, save_steps=5, save_path='outputs/ckpt')
    # trainer.save()









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


