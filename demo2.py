import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('demo2_config.yaml', task_dir='demo2_tasks')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()

    controller = palm.Controller(config='demo2_config.yaml', task_dir='demo2_tasks', for_train=False)
    controller.pred('mrqa', inference_model_dir='output_model/secondrun/infer_model')

