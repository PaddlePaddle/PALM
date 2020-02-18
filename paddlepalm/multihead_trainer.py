
from paddle import fluid
from paddle.fluid import layers
from paddlepalm.distribute import gpu_dev_count, cpu_dev_count, data_feeder, decode_fake
from paddlepalm import Trainer
from paddlepalm.utils import reader_helper
import numpy as np
import time
import sys

dev_count = 1 if gpu_dev_count <= 1 else gpu_dev_count
VERBOSE=False


class MultiHeadTrainer(Trainer):
    """
    The core unit to start a multi-task training/predicting session. A MultiHeadTrainer is built based on several Trainers. Beyond the inheritance of Trainer, it additionally achieves model backbone reuse across tasks, trainer sampling for multi-task learning, and multi-head inference for effective evaluation and prediction. 
    """
    
    def __init__(self, trainers):
        """Create a new multi_head_trainer.

        Args:
            trainers: a list of Trainer objects.

        """
        # if reuse_flags is not None:
        #     assert len(reuse_flags) == len(trainers)
        Trainer.__init__(self, '')

        self._trainers = trainers

        name_maxlen = max([len(i.name) for i in self._trainers])
        self._name_pads = {i.name: name_maxlen-len(i.name) for i in self._trainers}

        self._train_init = False
        self._predict_init = False
        self._feeded_var_names = None
        self._cur_train_step = 0
        self._target_vars = None

        self._inputname_to_varname = {}
        self._pred_input_name_list = []
        self._pred_input_varname_list = []
        self._pred_fetch_name_list = []
        self._pred_fetch_var_list = []

        self._exe = None

        self._save_protocol = {
            'input_names': 'self._pred_input_name_list',
            'input_varnames': 'self._pred_input_varname_list',
            'fetch_list': 'self._pred_fetch_name_list'}

        self._check_save = lambda: False
        for t in self._trainers:
            t._set_multitask()

    # def build_forward(self, backbone, heads):
    def build_forward(self):
        """
        Build forward computation graph for training, which usually built from input layer to loss node.

        Return:
            - loss_var: a Variable object. The computational graph variable(node) of loss.
        """
        head_dict = {}
        backbone = self._trainers[0]._backbone
        for i in self._trainers:
            assert i._task_head is not None and i._backbone is not None, "You should build forward for the {} task".format(i._name)
            assert i._backbone == backbone, "The backbone for each task must be the same"
            head_dict[i._name] = i._task_head
            
        train_prog = fluid.Program()
        train_init_prog = fluid.Program()
        self._train_prog = train_prog
        self._train_init_prog = train_init_prog

        def get_loss(i):
            head = head_dict[self._trainers[i].name]
            self._trainers[i]._lock_prog = True
            loss_var = self._trainers[i].build_forward(backbone, head)
            self._trainers[i]._lock_prog = False
            return loss_var
      
        task_fns = {i: lambda i=i: get_loss(i) for i in range(len(self._trainers))}

        with fluid.program_guard(train_prog, train_init_prog):
            task_id_var = fluid.data(name="__task_id",shape=[1],dtype='int64')

            loss_var = layers.switch_case(
                branch_index=task_id_var,
                branch_fns=task_fns
            )
        self._task_id_var = task_id_var
        self._loss_var = loss_var
        self._fetch_list = [loss_var.name]
        # for b in train_prog.blocks:
        #     for var in b.vars:
        #         pass
                # if 'task_id' in var:
                #     print(var)
                #     exit()
                # print(var)
        if not self._multi_task:
            self._init_exe_prog(for_train=True)
        return loss_var

    def fit_readers(self, reader_dict):
        raise NotImplementedError()

    def fit_readers_with_mixratio(self, readers, sampling_reference, num_epochs, phase='train'):
        """
        Bind readers and loaded train/predict data to trainers. The `num_epochs` argument only 
            works on `sampling_reference` task(trainer), and num_epochs of other tasks are infered from 
            their `mix_ratio`.

        Args:
            readers: a dict or list of Reader objects. For dict case, each key is a trainer's name, and the mapped value is the reader object to bind to the trainer. For list case, each 
            sampling_reference: a trainer name. The task(trainer) selected as baseline for task sampling. 
            num_epochs: training epochs of the sampling_reference task (trainer). 
        """
        self._check_phase(phase)

        if isinstance(readers, list):
            reader_dict = {k.name: v for k,v in zip(self._trainers, readers)}
        elif isinstance(readers, dict):
            reader_dict = readers
        else:
            raise ValueError()
        
        num_heads = len(self._trainers)
        assert len(reader_dict) == num_heads, "received number of readers is not consistent with trainers."

        trainer_dict = {t.name: t for t in self._trainers}
        assert sampling_reference in trainer_dict

        trainer_dict[sampling_reference]._set_task_id(self._task_id_var)
        trainer_dict[sampling_reference].fit_reader(reader_dict[sampling_reference])
        base_steps_pur_epoch = trainer_dict[sampling_reference]._steps_pur_epoch

        self._finish_steps = {}
        self._finish = {}
        input_names = []
        name_to_pos = []
        joint_shape_and_dtypes = []
        iterators = []
        prefixes = []
        mrs = []
        net_inputs = []
        global_steps = 0
        for t in self._trainers:
            assert t.name in reader_dict
            assert reader_dict[t.name].num_epochs is None, "{}: num_epochs is not None. \
                To run with multi-head mode, num_epochs of each Trainer should be set as None.".format(t.name)
            # print(num_epochs, t.mix_ratio, base_steps_pur_epoch)
            max_train_steps = int(num_epochs * t.mix_ratio * base_steps_pur_epoch)
            if not t._as_auxilary:
                print('{}: expected train steps {}.'.format(t.name, max_train_steps))
                sys.stdout.flush()
                self._finish_steps[t.name] = max_train_steps
                self._finish[t.name] = False
            else:
                self._finish_steps[t.name] = 9999999999
                self._finish[t.name] = True

            global_steps += max_train_steps
            if t.name != sampling_reference:
                t._set_task_id(self._task_id_var)
                t.fit_reader(reader_dict[t.name])
            net_inputs.append(t._net_inputs)
            prefixes.append(t.name)
            mrs.append(t.mix_ratio)
            iterators.append(t._raw_iterator_fn())
            input_names.append(t._input_names)
            name_to_pos.append(t._name_to_position)
            joint_shape_and_dtypes.append(t._shape_and_dtypes)

        print('Estimated overall train steps {}.'.format(global_steps))
        sys.stdout.flush()
        self._overall_train_steps = global_steps

        iterator_fn = reader_helper.create_multihead_iterator_fn(iterators, prefixes, joint_shape_and_dtypes, \
            mrs, input_names, name_to_pos, dev_count=dev_count)
        feed_batch_process_fn = reader_helper.create_feed_batch_process_fn(net_inputs)

        if gpu_dev_count > 1:
            distribute_feeder_fn = data_feeder(iterator_fn, feed_batch_process_fn, phase=phase, is_multi=True)
        else:
            distribute_feeder_fn = iterator_fn()

        if phase == 'train':
            self._train_reader = distribute_feeder_fn
            self._feed_batch_process_fn = feed_batch_process_fn
        elif phase == 'predict':
            self._predict_reader = distribute_feeder_fn
            self._pred_feed_batch_process_fn = feed_batch_process_fn

    def _check_finish(self, task_name, silent=False):
        trainers = {t.name:t for t in self._trainers}
        if trainers[task_name]._cur_train_step == self._finish_steps[task_name]:
            if not silent:
                print(task_name+' train finish!')
                sys.stdout.flush()
            self._finish[task_name]=True
        flags = list(set(self._finish.values()))
        return len(flags) == 1 and flags[0] == True
        
    def train(self, print_steps=5):
        """
        start training.

        Args:
            print_steps: int. Logging frequency of training message, e.g., current step, loss and speed.
        """
        iterator = self._train_reader
        self._distribute_train_prog = fluid.CompiledProgram(self._train_prog).with_data_parallel(loss_name=self._loss_var.name)
        for t in self._trainers:
            t._set_exe(self._exe)
            t._set_dist_train(self._distribute_train_prog)
            t._set_fetch_list(self._fetch_list)

        time_begin = time.time()
        for feed in iterator:
            # batch, task_id = feed
            rt_outputs, task_id = self.train_one_step(feed)

            task_rt_outputs = {k[len(self._trainers[task_id].name+'.'):]: v for k,v in rt_outputs.items() if k.startswith(self._trainers[task_id].name+'.')}
            self._trainers[task_id]._task_head.batch_postprocess(task_rt_outputs)
            if print_steps > 0 and self._cur_train_step % print_steps == 0:
                loss = rt_outputs[self._trainers[task_id].name+'.loss']
                loss = np.mean(np.squeeze(loss)).tolist()

                time_end = time.time()
                time_cost = time_end - time_begin

                print("global step: {}, {}: step {}/{} (epoch {}), loss: {:.3f}, speed: {:.2f} steps/s".format(
                       self._cur_train_step, ' '*self._name_pads[self._trainers[task_id].name]+self._trainers[task_id].name, \
                       (self._trainers[task_id]._cur_train_step-1) % self._trainers[task_id]._steps_pur_epoch + 1, \
                       self._trainers[task_id]._steps_pur_epoch, self._trainers[task_id]._cur_train_epoch, \
                       loss, print_steps / time_cost))
                sys.stdout.flush()
                time_begin = time.time()

            self._check_save()
            finish = self._check_finish(self._trainers[task_id].name)
            if finish:
                break

            # if cur_task.train_finish and cur_task.cur_train_step + cur_task.cur_train_epoch * cur_task.steps_pur_epoch == cur_task.expected_train_steps:
            #     print(cur_task.name+': train finished!')
            #     cur_task.save()

            # if (save_predict or save_ckpt) and self._cur_train_step % save_steps == 0:
            #     if save_predict:
            #         self.save(save_path, suffix='pred.step'+str(self._cur_train_step))
            #     if save_ckpt:
            #         fluid.io.save_persistables(self._exe, os.path.join(save_path, 'ckpt.step'+str(self._cur_train_step)), self._train_prog)
            #         print('checkpoint has been saved at '+os.path.join(save_path, 'ckpt.step'+str(self._cur_train_step)))


    def train_one_step(self, batch):

        if dev_count > 1:
            assert isinstance(batch, tuple)
            task_id = batch[0][0]['__task_id'][0]
        else:
            assert isinstance(batch, dict)
            task_id = batch['__task_id'][0]
            
        # rt_outputs = self._trainers[task_id].train_one_step(batch, self._exe, self._distribute_train_prog, self._fetch_list)
        rt_outputs = self._trainers[task_id].train_one_step(batch)

        self._cur_train_step += 1
        self._check_save()
        return rt_outputs, task_id
        
        # if dev_count > 1:
        #     # feed, mask, task_id = batch
        #     for f in feed:
        #         f['branch'] = np.array([task_id], dtype='int64')
        #     rt_outputs = self.exe.run(self._distribute_train_prog, feed=feed, fetch_list=self._trainers[task_id]._fetch_list)
        #     num_fakes = decode_fake(len(rt_outputs[0]), mask, self._trainers[task_id]._batch_size)
        #     for _ in range(num_fakes):
        #         for item in rt_outputs:
        #             item.pop()
        # else:
        #     feed, task_id = batch
        #     feed['branch'] = np.array([task_id], dtype='int64')
        #     rt_outputs = self._exe.run(self._distribute_train_prog, feed=feed, fetch_list=self._trainers[task_id]._fetch_list)

    def predict_one_batch(self, batch):
        raise NotImplementedError()

    def predict(self, output_dir=None, print_steps=1000):
        raise NotImplementedError()

    @property
    def overall_train_steps(self):
        return self._overall_train_steps
