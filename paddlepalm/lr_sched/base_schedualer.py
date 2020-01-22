
class Schedualer():

    def __init__(self):
        self._prog = None
    
    def _set_prog(self, prog):
        self._prog = prog

    def _build(self, learning_rate):
        raise NotImplementedError()

