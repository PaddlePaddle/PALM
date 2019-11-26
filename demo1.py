import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config_demo1.yaml', task_dir='demo1_tasks')
    controller.load_pretrain('pretrain_model/bert/params')
    controller.train()

