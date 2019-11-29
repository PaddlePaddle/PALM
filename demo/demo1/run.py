import paddlepalm as palm

if __name__ == '__main__':
    controller = palm.Controller('config.yaml')
    controller.load_pretrain('../../pretrain/bert-en-uncased-large/params')
    controller.train()

