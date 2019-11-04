import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config_demo3.yaml', task_dir='demo3_tasks')
    controller.load_pretrain('pretrain_model/ernie/params')
    controller.train()

