
class BaseOptimizer():

    def __init__(self, loss_var, lr, lr_schedualer=None):
        self._prog = None
        self._lr_schedualer = lr_schedualer

    def build(self, grad_clip=None):
        pass

    def _set_prog(self, prog, init_prog):
        self._prog = prog
        self._init_prog = prog
        if self._lr_schedualer is not None:
            self._lr_schedualer._set_prog(prog)

    def get_cur_learning_rate(self):
        pass


