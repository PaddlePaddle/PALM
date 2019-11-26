import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config.yaml', task_dir='tasks')
    controller.load_pretrain('../../pretrain_model/ernie/params')
    controller.train()

