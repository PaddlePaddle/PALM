import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config_demo3.yaml', task_dir='demo3_tasks')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()

    controller = palm.Controller(config='config_demo3.yaml', task_dir='demo3_tasks', for_train=False)
    controller.pred('cls4mrqa', inference_model_dir='output_model/thirdrun/infer_model')

