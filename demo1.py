import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config_demo1.yaml', task_dir='demo1_tasks')
    controller.load_pretrain('pretrain_model/bert/params')
    controller.train()

    controller = palm.Controller(config='config_demo1.yaml', task_dir='demo1_tasks', for_train=False)
    controller.pred('mrqa', inference_model_dir='output_model/firstrun/infer_model')

