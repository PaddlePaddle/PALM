
from paddlepalm.lr_sched.base_schedualer import Schedualer
import paddle.fluid as fluid

def WarmupSchedualer(Schedualer):
    """ Applies linear warmup of learning rate from 0 to learning_rate until warmup_steps, and then decay to 0 linearly until num_train_steps."""

    def __init__(self, warmup_steps):
        schedualer.__init__(self)
        self.warmup_steps = warmup_steps

    def _build(self, learning_rate):

        with self._prog._lr_schedule_guard():
            lr = fluid.layers.tensor.create_global_var(
                shape=[1],
                value=0.0,
                dtype='float32',
                persistable=True,
                name="scheduled_learning_rate")

            global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(global_step < self.warmup_steps):
                    warmup_lr = learning_rate * (global_step / self.warmup_steps)
                    fluid.layers.tensor.assign(warmup_lr, lr)
                with switch.default():
                    fluid.layers.tensor.assign(learning_rate, lr)

            return lr

