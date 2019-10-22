import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('demo1_config.yaml', task_dir='demo1_tasks')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()

    controller = palm.Controller(config='demo1_config.yaml', task_dir='demo1_tasks', for_train=False)
    controller.pred('mrqa', inference_model_dir='output_model/firstrun/infer_model')

