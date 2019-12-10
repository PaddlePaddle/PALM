import paddlepalm as palm

if __name__ == '__main__':

    match_reader = palm.reader.match(train_file, file_format='csv', tokenizer='wordpiece', lang='en')
    mrc_reader = palm.reader.mrc(train_file, phase='train')
    mlm_reader = palm.reader.mlm(train_file, phase='train')
    palm.reader.

    match = palm.tasktype.cls(num_classes=4)
    mrc = palm.tasktype.match(learning_strategy='pairwise')
    mlm = palm.tasktype.mlm()
    mlm.print()

    
    bb_flags = palm.load_json('./pretrain/ernie/ernie_config.json')
    bb = palm.backbone.ernie(bb_flags['xx'], xxx)
    bb.print()

    match4mrqa = palm.Task('match4mrqa', match_reader, match_tt)
    mrc4mrqa = palm.Task('match4mrqa', match_reader, match_tt)

    # match4mrqa.reuse_with(mrc4mrqa)


    controller = palm.Controller([mrqa, match4mrqa, mlm4mrqa])

    loss = controller.build_forward(bb, mask_task=[])

    n_steps = controller.estimate_train_steps(basetask=mrqa, num_epochs=2, batch_size=8, dev_count=4)
    adam = palm.optimizer.Adam(loss)
    sched = palm.schedualer.LinearWarmup(learning_rate, max_train_steps=n_steps, warmup_steps=0.1*n_steps)
    
    controller.build_backward(optimizer=adam, schedualer=sched, weight_decay=0.001, use_ema=True, ema_decay=0.999)

    controller.random_init_params()
    controller.load_pretrain('../../pretrain_model/ernie/params')
    controller.train()





    # controller = palm.Controller(config='config.yaml', task_dir='tasks', for_train=False)
    # controller.pred('mrqa', inference_model_dir='output_model/secondrun/mrqa/infer_model')


