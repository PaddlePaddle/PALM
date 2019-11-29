import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config.yaml', task_dir='tasks')
    controller.load_pretrain('../../pretrain/ernie-en-uncased-large/params')
    controller.train()

    controller = palm.Controller(config='config.yaml', task_dir='tasks', for_train=False)
    controller.pred('mrqa', inference_model_dir='output_model/secondrun/mrqa/infer_model')


