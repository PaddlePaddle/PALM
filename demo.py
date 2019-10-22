import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config.yaml', task_dir='task_instance')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()

    controller = palm.Controller(config='config.yaml', task_dir='task_instance', for_train=False)
    controller.pred('mrqa', inference_model_dir='output_model/firstrun/infer_model')

