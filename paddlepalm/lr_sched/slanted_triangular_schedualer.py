from paddlepalm.lr_sched.base_schedualer import Schedualer
from paddle import fluid

class TriangularSchedualer(Schedualer):

    """ Implementation of Slanted Triangular learning rate schedual method, more details refer to https://arxiv.org/pdf/1801.06146.pdf . Apply linear warmup of learning rate from 0 to learning_rate until warmup_steps, and then decay to 0 linearly until num_train_steps."""

    def __init__(self, warmup_steps, num_train_steps):
        """Create a new TriangularSchedualer object.

        Args:
            warmup_steps: the learning rate will grow from 0 to max_learning_rate over `warmup_steps` steps.
            num_train_steps: the number of train steps.

        """
        Schedualer.__init__(self)
        assert num_train_steps > warmup_steps > 0
        self.warmup_steps = warmup_steps
        self.num_train_steps = num_train_steps
        

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
                    decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                        learning_rate=learning_rate,
                        decay_steps=self.num_train_steps,
                        end_learning_rate=0.0,
                        power=1.0,
                        cycle=False)
                    fluid.layers.tensor.assign(decayed_lr, lr)

            return lr


