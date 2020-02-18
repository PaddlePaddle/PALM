# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
import json
from paddle import fluid
import time
import sys
import numpy as np
import paddlepalm.utils.basic_helper as helper
from paddlepalm.utils import reader_helper, saver
from paddlepalm.distribute import gpu_dev_count, data_feeder, decode_fake
# from paddlepalm.default_settings import *

DEBUG=False


class Trainer(object):
    """
    The core unit to start a training/predicting session for single task. A trainer is to build computation graph, manage training and evaluation process, achieve model/checkpoint saving and pretrain_model/checkpoint loading.
    """

    def __init__(self, name, mix_ratio=1.0, reuse_head_with=None):
        """Create a new trainer.

        Args:
            name: string. The name of the trainer(training task).
            mix_ratio: sampling weight of this trainer in multi-task learning mode. Default is 1.0.
            reuse_head_with: reuse parameters of task head with another trainer. Default is None, not reuse with others.

        """

        self._name = name
        self._pred_reader = None
        self._task_head = None
        self._pred_head = None
      
        self._train_reader = None
        self._predict_reader = None
        self._train_iterator = None
        self._predict_iterator = None

        self._train_init = False
        self._predict_init = False
        self._train_init_prog = None
        self._pred_init_prog = None

        self._check_save = lambda: False

        # if save_predict_model:
        #     self._save_predict_model = True
        #     assert pred_head is not None, "pred_head is required to save predict model."
        #     self._pred_reader = reader.clone(phase='predict')
        # else:
        #     assert pred_head is None, "You should set save_predict_model as True, or the pred_head is invalid." 
        #     self._save_predict_model = False
        #     self._pred_reader = None

        # self._save_steps = save_steps

        self._task_reuse_scope = name if reuse_head_with is None else reuse_head_with

        self._feeded_var_names = None
        self._target_vars = None

        self._num_examples = 0

        self._multi_task = False
        self._as_auxilary = False
        self._task_id = None

        # training process management
        self._mix_ratio = mix_ratio
        self._expected_train_steps = None
        self._expected_train_epochs = None
        self._steps_pur_epoch = None
        self._pred_steps_pur_epoch = None
        self._cur_train_epoch = 0
        self._cur_train_step = 0
        self._train_finish = False

        self._inputname_to_varname = {}
        self._pred_input_name_list = []
        self._pred_input_varname_list = []
        self._pred_fetch_name_list = []
        self._pred_fetch_var_list = []

        # exe is built when random_init_params is called.
        # self._exe = helper.build_executor(gpu_dev_count>0)
        self._exe = None

        self._save_protocol = {
            'input_names': 'self._pred_input_name_list',
            'input_varnames': 'self._pred_input_varname_list',
            'fetch_list': 'self._pred_fetch_name_list'}

        self._lock = False
        self._lock_prog = False
        self._build_forward = False

    def build_forward(self, backbone, task_head):
        """
        Build forward computation graph for training, which usually built from input layer to loss node.

        Args:
            backbone: a Backbone object with phase == 'train', which is used to extract multi-level text features, e.g., contextual word embedding and sentence embedding.
            head: a Head object with phase == 'train', which is used to build task specific output layers.
        
        Return:
            loss_var: a Variable object. The computational graph variable(node) of loss.
        """


        # assert not self._multi_task, "you cannot build_forward in trainer when a train is wrapper by MultiHeadTrainer."
        self._task_head = task_head
        self._backbone = backbone

        # assert self._backbone is not None, "backbone is required for Trainer to build net forward to run with single task mode"
        self._build_forward = True
        
        # create reader, task
        # then check i/o across reader, backbone and task_layer
        task_attrs = []
        pred_task_attrs = []

        task_attr_from_reader = helper.encode_inputs(self._task_head.inputs_attrs['reader'], self.name)
        # task_attr_from_reader = self._task_head.inputs_attrs['reader']

        # _check_io(backbone.inputs_attr, inst._reader['train'].outputs_attr, in_name=bb_name+'_backbone', out_name='reader.train')
        # _check_io(inst.taskblock['train'].inputs_attrs['reader'], inst._reader['train'].outputs_attr, in_name='task_paradigm.train.reader', out_name='reader.train')
        # _check_io(inst._taskblock['train'].inputs_attrs['backbone'], train_backbone.outputs_attr, in_name='task_paradigm.train.backbone', out_name=bb_name+'_backbone')


        # merge reader input attrs from backbone and task_instances
        input_names, shape_and_dtypes, name_to_position = reader_helper.merge_input_attrs(backbone.inputs_attr, task_attr_from_reader, insert_taskid=False)
        # shapes: [task_id, shapes_of_backbone, shapes_of_inst1, ..., shapes_of_instN]
        self._shape_and_dtypes = shape_and_dtypes
        self._name_to_position = name_to_position
        self._input_names = input_names

        if DEBUG:
            print('----- for debug -----')
            print('joint input names:')
            print(joint_input_names)
            print('joint input shape and dtypes:')
            print(joint_shape_and_dtypes)

        input_attrs = [[i, j, k] for i, (j,k) in zip(input_names, shape_and_dtypes)]

        train_prog = fluid.Program()
        train_init_prog = fluid.Program()

        if not self._lock_prog:
            self._train_prog = train_prog
            self._train_init_prog = train_init_prog

        if not self._lock_prog:
            with fluid.program_guard(train_prog, train_init_prog):
                net_inputs = reader_helper.create_net_inputs(input_attrs, is_async=False)
                bb_output_vars = backbone.build(net_inputs)
        else:
            net_inputs = reader_helper.create_net_inputs(input_attrs, is_async=False)
            bb_output_vars = backbone.build(net_inputs)
        self._net_inputs = net_inputs
        assert sorted(bb_output_vars.keys()) == sorted(backbone.outputs_attr.keys())

        # self._bb_output_vars.keys

        # fluid.framework.switch_main_program(train_prog)
        # fluid.framework.switch_startup_program(train_init_prog)

        task_output_vars = {}
        task_inputs = {'backbone': bb_output_vars}
        task_inputs_from_reader = helper.decode_inputs(net_inputs, self.name)
        task_inputs['reader'] = task_inputs_from_reader

        scope = self.name+'.'
        if not self._lock_prog:
            with fluid.program_guard(train_prog, train_init_prog):
                with fluid.unique_name.guard(scope):
                    output_vars = self._build_head(task_inputs, phase='train', scope=scope)
        else:
            with fluid.unique_name.guard(scope):
                output_vars = self._build_head(task_inputs, phase='train', scope=scope)

        output_vars = {self.name+'.'+key: val for key, val in output_vars.items()}
        old = len(task_output_vars) # for debug
        task_output_vars.update(output_vars)
        assert len(task_output_vars) - old == len(output_vars) # for debug

        bb_fetches = {k: v.name for k,v in bb_output_vars.items()}
        task_fetches = {k: v.name for k,v in task_output_vars.items()}
        self._fetches = task_fetches
        self._fetch_names, self._fetch_list = zip(*self._fetches.items())
        # fetches = task_fetches
        # fetches['__task_id'] = net_inputs['__task_id'].name

        # compute loss
        # task_id_var = net_inputs['__task_id']
        # task_id_vec = layers.one_hot(task_id_var, num_instances)
        # losses = fluid.layers.concat([task_output_vars[inst.name+'/loss'] for inst in instances], axis=0)
        # loss = layers.reduce_sum(task_id_vec * losses)
        if not self._lock_prog:
            with fluid.program_guard(train_prog, train_init_prog):
                loss_var = fluid.layers.reduce_sum(task_output_vars[self.name+'.loss'])
        else:
            loss_var = fluid.layers.reduce_sum(task_output_vars[self.name+'.loss'])

        # for _id, block in enumerate(self._train_prog.blocks):
        #   for var in block.vars:
        #     print("[debug] : %d, %s" % (_id, var))
        self._loss_var = loss_var

        if not self._multi_task:
            self._init_exe_prog(for_train=True)

        return loss_var

    def build_predict_forward(self, pred_backbone, pred_head):
        """
        Build computation graph for evaluation and prediction.

        Arguments:
            - pred_backbone: a Backbone object with phase == 'predict'. For evaluating model during training, the predict backbone should keep the same with train backbone.
            - pred_head: a Head object with phase == 'predict'. For evaluating model during training, the predict head should keep the same with train head.
        
        Return:
            - output_vars: dict type. Each value is a computational graph variable(node) argumented by pred_head outputs_attr.
        """
        self._pred_head = pred_head
        self._pred_backbone = pred_backbone
        # self._pred_reader = self._reader.clone(phase='pred')
        pred_task_attr_from_reader = helper.encode_inputs(self._pred_head.inputs_attrs['reader'], self.name)
        # pred_task_attr_from_reader = self._pred_head.inputs_attrs['reader']

        # _check_io(pred_backbone.inputs_attr, pred_reader.outputs_attr, in_name=bb_name+'_backbone', out_name='reader.pred')

        # _check_io(pred_backbone.inputs_attr, pred_reader.outputs_attr, in_name=bb_name+'_backbone', out_name='reader.pred')
        # _check_io(pred_parad.inputs_attrs['reader'], pred_reader.outputs_attr, in_name='task_paradigm.pred.reader', out_name='reader.pred')
        # _check_io(pred_parad.inputs_attrs['backbone'], pred_backbone.outputs_attr, in_name='task_paradigm.pred.backbone', out_name=bb_name+'_backbone')
        pred_input_names, pred_shape_and_dtypes, pred_name_to_position = reader_helper.merge_input_attrs(pred_backbone.inputs_attr, pred_task_attr_from_reader, insert_taskid=False)
        pred_input_attrs = [[i, j, k] for i, (j,k) in zip(pred_input_names, pred_shape_and_dtypes)]
        self._pred_shape_and_dtypes = pred_shape_and_dtypes
        self._pred_name_to_position = pred_name_to_position

        pred_prog = fluid.Program()
        self._pred_prog = pred_prog
        pred_init_prog = fluid.Program()
        self._pred_init_prog = pred_init_prog
        with fluid.program_guard(pred_prog, pred_init_prog):
            pred_net_inputs = reader_helper.create_net_inputs(pred_input_attrs)
            # pred_bb_output_vars = pred_backbone.build(pred_net_inputs, scope_name='__paddlepalm_')
            pred_bb_output_vars = pred_backbone.build(pred_net_inputs)
            self._pred_net_inputs = pred_net_inputs

        # prepare predict vars for saving inference model
        with fluid.program_guard(pred_prog, pred_init_prog):
            cur_inputs = helper.decode_inputs(pred_net_inputs, self.name)
            # self.pred_input = cur_inputs
            self._pred_input_name_list, self._pred_input_varname_list = \
                zip(*[[k, v.name] for k,v in cur_inputs.items()])

            pred_task_inputs = {'backbone': pred_bb_output_vars, 'reader': cur_inputs}
            scope = self.name + '.'
            with fluid.unique_name.guard(scope):
                output_vars = self._build_head(pred_task_inputs, phase='predict', scope=scope)

        if output_vars is not None:
            self._pred_fetch_name_list, self._pred_fetch_list = zip(*output_vars.items())
        else:
            self._pred_fetch_name_list = []
            self._pred_fetch_var_list = []

        if not self._multi_task:
            self._init_exe_prog(for_train=False)
            self._exe.run(self._pred_init_prog)
            
        return output_vars

    def build_backward(self, optimizer, weight_decay=None, use_ema=False, ema_decay=None):
        """
        Build backward computation graph and training strategy.

        Arguments:
            - optimizer: 
            - weight_decay: optional, default is None (disable weight decay).
            - use_ema: optional, default is False. The flag to control whether to apply Exponential Moving Average strategy on parameter updates.
            - ema_decay: optional, default is None. Only works with use_ema == True. Control decay rate of EMA strategy.

        """
        # assert not self._multi_task, "you cannot build_backward in trainer when a train is wrapper by MultiHeadTrainer."
        # build optimizer
        assert self._loss_var is not None and self._train_init_prog is not None, "train graph not foung! You should build_forward first."
        optimizer._set_prog(self._train_prog, self._train_init_prog)
        with fluid.program_guard(self._train_prog, self._train_init_prog):
            param_grads = optimizer._build()

            if weight_decay is not None:

                param_list = dict()

                for param in self._train_prog.global_block().all_parameters():
                    param_list[param.name] = param * 1.0
                    param_list[param.name].stop_gradient = True

                def exclude_from_weight_decay(name):
                    if name.find("layer_norm") > -1:
                        return True
                    bias_suffix = ["_bias", "_b", ".b_0"]
                    for suffix in bias_suffix:
                        if name.endswith(suffix):
                            return True
                    return False

                for param, grad in param_grads:
                    if exclude_from_weight_decay(param.name):
                        continue
                    with param.block.program._optimized_guard(
                        [param, grad]), fluid.framework.name_scope("weight_decay"):
                        updated_param = param - param_list[
                            param.name] * weight_decay * optimizer.get_cur_learning_rate()
                        fluid.layers.assign(output=param, input=updated_param)


            # loss.persistable = True
            if use_ema:
                ema = fluid.optimizer.ExponentialMovingAverage(ema_decay)
                ema.update()

        # for bid, block in enumerate(self._train_prog.blocks):
        #     print('block id: '+str(bid))
        #     for var in block.vars:
        #         print("%d : %s" % (bid, var))
            
        # print(self._train_prog)
        self._exe.run(self._train_init_prog)

    def set_as_aux(self):
        """Set the task in this trainer as auxilary task. \nCAUSIOUS: This API only works on multi-task learning mode. Each task is set as target task by default. """
        self._as_auxilary = True

    def fit_reader(self, reader, phase='train'):
        """
        Bind a reader and loaded train/predict data to trainer. 
        
        Args:
            reader: a Reader object. The running phase of the reader should be consistent with `phase` argument of this method.
            phase: running phase. Currently support: train, predict.

        """
        # assert not self._multi_task, "you cannot fit_reader in trainer when a train is wrapper by MultiHeadTrainer."
        # load data

        self._check_phase(phase)
        if phase=='train':
            assert self._shape_and_dtypes is not None, "You need to build_forward or build_predict_head first to prepare input features."
        else:
            assert self._pred_shape_and_dtypes is not None, "You need to build_forward     or build_predict_head first to prepare input features."

        # 这里不确定是否要向上取整，需确认
        # tail = self._num_examples % batch_size > 0
        # self._steps_pur_epoch = self._num_examples // batch_size + 1 if tail else 0
        
        batch_size = reader._batch_size

        self._num_epochs = reader.num_epochs
        if phase == 'train':
            self._train_reader = reader
            self._steps_pur_epoch = reader.num_examples // batch_size
            shape_and_dtypes = self._shape_and_dtypes
            name_to_position = self._name_to_position
            if self._task_id is not None:
                self._net_inputs['__task_id'] = self._task_id
            net_inputs = self._net_inputs
            self._train_batch_size = batch_size
            self._num_examples = reader.num_examples
            reader_helper.check_io(self._backbone.inputs_attr, reader.outputs_attr, in_name='backbone', out_name='reader(train)')
            reader_helper.check_io(self._task_head.inputs_attrs['reader'], reader.outputs_attr, in_name='task_head(reader)', out_name='reader(train)')
            reader_helper.check_io(self._task_head.inputs_attrs['backbone'], self._backbone.outputs_attr, in_name='task_head(backbone, train)', out_name='backbone')
        elif phase == 'predict':
            self._predict_reader = reader
            # tail = self._num_examples % batch_size > 0
            # self._pred_steps_pur_epoch = reader.num_examples // batch_size + 1 if tail else 0
            self._pred_steps_pur_epoch = reader.num_examples // batch_size 
            shape_and_dtypes = self._pred_shape_and_dtypes
            name_to_position = self._pred_name_to_position
            net_inputs = self._pred_net_inputs
            self._predict_batch_size = batch_size
            self._pred_num_examples = reader.num_examples
            reader_helper.check_io(self._pred_backbone.inputs_attr, reader.outputs_attr, in_name='backbone', out_name='reader(predict)')
            reader_helper.check_io(self._pred_head.inputs_attrs['reader'], reader.outputs_attr, in_name='task_head(reader)', out_name='reader(predict)')
            reader_helper.check_io(self._pred_head.inputs_attrs['backbone'], self._pred_backbone.outputs_attr, in_name='task_head(backbone, predict)', out_name='backbone')
        else:
            raise NotImplementedError()
            
        print('ok!')

        # merge dataset iterators and create net input vars
        iterator = reader._iterator()
        prefix = self.name


        # merge dataset iterators and create net input vars
        iterator = reader._iterator()
        prefix = self.name

        # 对yield出的数据进行runtime检查和适配
        iterator_fn = reader_helper.create_iterator_fn(iterator, prefix, shape_and_dtypes, name_to_position, return_type='dict')
        self._raw_iterator_fn = iterator_fn
        feed_batch_process_fn = reader_helper.create_feed_batch_process_fn(net_inputs)
        if gpu_dev_count > 1:
            distribute_feeder_fn = data_feeder(iterator_fn, feed_batch_process_fn, phase=phase)
        else:
            distribute_feeder_fn = iterator_fn()

        if phase == 'train':
            self._train_iterator = distribute_feeder_fn
            self._feed_batch_process_fn = feed_batch_process_fn
        elif phase == 'predict':
            self._predict_iterator = distribute_feeder_fn
            self._pred_feed_batch_process_fn = feed_batch_process_fn
        # return distribute_feeder_fn()


    def load_ckpt(self, model_path):
        """
        load training checkpoint for further training or predicting.

        Args:
            model_path: the path of saved checkpoint/parameters.
        """
        # load pretrain model (or ckpt)
        # assert self._exe is not None, "You need to random_init_params before load checkpoints."
        # if phase == 'train' and not self._train_init:
        #     self._init_exe_prog(for_train=True)
        #     self._exe.run(self._train_init_prog)
        # if phase == 'predict' and not self._predict_init:
        #     self._init_exe_prog(for_train=False)
        #     self._exe.run(self._pred_init_prog)

        assert self._train_init_prog is not None or self._pred_init_prog is not None, "model graph not built. You should at least build_forward or build_predict_forward to load its checkpoint."

        # if phase == 'train':
        #     assert self._train_init_prog is not None, "train graph not found! You should build_forward first before load checkpoint."
        if self._train_init_prog is not None:
            saver.init_pretraining_params(
                self._exe,
                model_path,
                convert=False,
                main_program=self._train_init_prog,
                strict=True)
        # elif phase == 'predict':
        elif self._pred_init_prog is not None:
            # assert self._pred_init_prog is not None, "predict graph not found! You should build_predict_head first before load checkpoint."
            saver.init_pretraining_params(
                self._exe,
                model_path,
                convert=False,
                main_program=self._pred_init_prog,
                strict=True)
        else:
            raise Exception("model not found. You should at least build_forward or build_predict_forward to load its checkpoint.")

    def load_predict_model(self, model_path, convert=False):
        """
        load pretrain models(backbone) for training.

        Args:
            model_path: the path of saved pretrained parameters.
        """

        assert self._pred_prog is not None, "training graph not found. You should at least build_forward to load its pretrained parameters."

        saver.init_pretraining_params(
            self._exe,
            model_path,
            convert=convert,
            main_program=self._pred_prog)
        # raise NotImplementedError()

    def load_pretrain(self, model_path, convert=False):
        """
        load pretrain models(backbone) for training.

        Args:
            model_path: the path of saved pretrained parameters.
        """
        # load pretrain model (or ckpt)
        # assert self._exe is not None, "You need to random_init_params before load pretrain models."
        assert self._train_init_prog is not None, "training graph not found. You should at least build_forward to load its pretrained parameters."

        saver.init_pretraining_params(
            self._exe,
            model_path,
            convert=convert,
            main_program=self._train_init_prog)

    def set_saver(self, save_path, save_steps, save_type='ckpt'):
        """
        create a build-in saver into trainer. A saver will automatically save checkpoint or predict model every `save_steps` training steps.

        Args:
            save_path: a string. the path to save checkpoints or predict models.
            save_steps: an integer. the frequency to save models.
            save_type: a string. The type of saved model. Currently support checkpoint(ckpt) and predict model(predict), default is ckpt. If both two types are needed to save, you can set as "ckpt,predict".

        """
        

        save_type = save_type.split(',')
        if 'predict' in save_type:
            assert self._pred_head is not None, "Predict head not found! You should build_predict_head first if you want to save predict model."
            assert save_path is not None and save_steps is not None, 'save_path and save_steps is required to save model.'
            self._save_predict = True
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            self._save_predict = False

        if 'ckpt' in save_type:
            if save_path is not None and save_steps is not None:
                self._save_ckpt = True
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            else:
                "WARNING: save_path or save_steps is not set, model will not be saved during training."
                self._save_ckpt = False
        else:
            self._save_ckpt = False

        def temp_func():
            if (self._save_predict or self._save_ckpt) and self._cur_train_step % save_steps == 0:

                if self._save_predict:
                    self._save(save_path, suffix='pred.step'+str(self._cur_train_step))
                    print('predict model has been saved at '+os.path.join(save_path, 'pred.step'+str(self._cur_train_step)))
                    sys.stdout.flush()
                if self._save_ckpt:
                    fluid.io.save_persistables(self._exe, os.path.join(save_path, 'ckpt.step'+str(self._cur_train_step)), self._train_prog)
                    print('checkpoint has been saved at '+os.path.join(save_path, 'ckpt.step'+str(self._cur_train_step)))
                    sys.stdout.flush()
                return True
            else:
                return False

        self._check_save = temp_func
            
    def train(self, print_steps=5):
        """
        start training.

        Args:
            print_steps: int. Logging frequency of training message, e.g., current step, loss and speed.
        """
        
        iterator = self._train_iterator
        self._distribute_train_prog = fluid.CompiledProgram(self._train_prog).with_data_parallel(loss_name=self._loss_var.name)

        # if save_path is not None or save_steps is not None:
        #     assert self._save_predict_model, "If you want to save model, you need set save_predict_model=True when this trainer is built."
        # if self._save_predict_model:
        #     if save_path is None and save_steps is None:
        #         print('Warning: model will not be saved for this run. If you want to save model, set save_path and save_steps.')
        #     else:
        #         assert save_path is not None, "argument save_path is required to save models."
        #         assert save_steps == -1 or save_steps > 0, "argument save_steps should be -1 (only save the last step of this task) or larger than 0"
        #         if save_path is not None and not os.path.exists(save_path):
        #             os.makedirs(save_path)
        # else:
        #     assert save_path is None, "You should set save_predict_model as True, or the argument save_path is invalid."
        #     assert save_steps is None, "You should set save_predict_model as True, or the argument save_steps is invalid."

        time_begin = time.time()
        for feed in iterator:
            rt_outputs = self.train_one_step(feed)
            # if gpu_dev_count > 1:
            #     feed, mask = feed
            # rt_outputs = self._exe.run(self._train_prog, feed=feed, fetch_list=self._fetch_list)
            # print(rt_outputs)
            # print(len(rt_outputs))
            # if gpu_dev_count > 1:
            #     while mask.pop() == False:
            #         rt_outputs.pop()

            # rt_outputs = {k:v for k,v in zip(self._fetch_names, rt_outputs)}

            task_rt_outputs = {k[len(self.name+'.'):]: v for k,v in rt_outputs.items() if k.startswith(self.name+'.')}
            self._task_head.batch_postprocess(task_rt_outputs)


            if print_steps > 0 and self._cur_train_step % print_steps == 0:
                loss = rt_outputs[self.name+'.loss']
                loss = np.mean(np.squeeze(loss)).tolist()

                time_end = time.time()
                time_cost = time_end - time_begin

                print("step {}/{} (epoch {}), loss: {:.3f}, speed: {:.2f} steps/s".format(
                       (self._cur_train_step-1) % self._steps_pur_epoch + 1 , self._steps_pur_epoch, self._cur_train_epoch,
                       loss, print_steps / time_cost))
                sys.stdout.flush()
                time_begin = time.time() 
                # self._check_save()
            # if cur_task.train_finish and cur_task.cur_train_step + cur_task.cur_train_epoch * cur_task.steps_pur_epoch == cur_task.expected_train_steps:
            #     print(cur_task.name+': train finished!')
            #     cur_task.save()

            if self._num_epochs is None and not self._multi_task and self._cur_train_step == self._steps_pur_epoch:
                break
        # save_path = os.path.join(main_conf['save_path'], 'ckpt',
        #                          "step_" + str(global_step))
        # fluid.io.save_persistables(self.exe, save_path, saver_program)

        # print("ALL tasks train finished, exiting...")
        
    def predict(self, output_dir=None, print_steps=1000):
        """
        start predicting.

        Args:
            output_dir: str. The path to save prediction results, default is None. If set as None, the results would output to screen directly. 
            print_steps: int. Logging frequency of predicting message, e.g., current progress and speed.
        """
        iterator = self._predict_iterator
        self._distribute_pred_prog = fluid.CompiledProgram(self._pred_prog).with_data_parallel()

        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        time_begin = time.time()
        
        cur_predict_step = 0
        for feed in iterator:
            rt_outputs = self.predict_one_batch(feed)
            # rt_outputs = {k[len(self.name+'.'):]: v for k,v in rt_outputs.items() if k.startswith(self.name+'.')}
            self._pred_head.batch_postprocess(rt_outputs)

            cur_predict_step += 1

            if print_steps > 0 and cur_predict_step % print_steps == 0:
                time_end = time.time()
                time_cost = time_end - time_begin

                print("batch {}/{}, speed: {:.2f} steps/s".format(
                       cur_predict_step, self._pred_steps_pur_epoch,
                       print_steps / time_cost))
                sys.stdout.flush()
                time_begin = time.time()

        if self._pred_head.epoch_inputs_attrs:
            reader_outputs = self._predict_reader.get_epoch_outputs()
        else:
            reader_outputs = None

        results = self._pred_head.epoch_postprocess({'reader':reader_outputs}, output_dir=output_dir)
        return results

    def _check_phase(self, phase):
        assert phase in ['train', 'predict'], "Supported phase: train, predict,"

    def _set_multitask(self):
        self._multi_task = True

    def _set_task_id(self, task_id):
        self._task_id = task_id

    def _init_exe_prog(self, for_train=True):
        if not self._train_init and not self._predict_init:
            on_gpu = gpu_dev_count > 0
            self._exe = helper.build_executor(on_gpu)

        if for_train:
            assert self._train_prog is not None, "train graph not found! You should build_forward first before you random init parameters."
            self._train_init = True
        else:
            assert self._pred_prog is not None, "predict graph not found! You should build_predict_head first before you random init parameters."
            self._predict_init = True

    # def random_init_params(self):
    #     """
    #     randomly initialize model parameters.
    #     """
    #     
    #     if not self._train_init:
    #         self._init_exe_prog()
    #     
    #     print('random init params...')
    #     self._exe.run(self._train_init_prog)

    def get_one_batch(self, phase='train'):
        self._check_phase(phase)
        if phase == 'train':
            return next(self._train_reader)
        elif phase == 'predict':
            return next(self._predict_reader)
        else:
            raise NotImplementedError()

    def _set_exe(self, exe):
        self._exe = exe

    def _set_dist_train(self, prog):
        self._distribute_train_prog = prog

    def _set_fetch_list(self, fetch_list):
        self._fetch_list = fetch_list

    # def train_one_step(self, batch, executor=None, distribute_train_prog=None, fetch_list=None):
    def train_one_step(self, batch):
        # exe = self._exe if executor is None else executor
        # distribute_train_prog = self._distribute_train_prog if distribute_train_prog is None else distribute_train_prog
        # fetch_list = self._fetch_list if fetch_list is None else fetch_list

        exe = self._exe
        distribute_train_prog = self._distribute_train_prog
        fetch_list = self._fetch_list

        if gpu_dev_count > 1:
            feed, mask = batch
            rt_outputs = exe.run(distribute_train_prog, feed=feed, fetch_list=fetch_list)
            num_fakes = decode_fake(len(rt_outputs[0]), mask, self._train_batch_size)
            if num_fakes:
                rt_outputs = [i[:-num_fakes] for i in rt_outputs]
        
        else:
            feed = self._feed_batch_process_fn(batch)
            rt_outputs = exe.run(distribute_train_prog, feed=feed, fetch_list=fetch_list)

        rt_outputs = {k:v for k,v in zip(self._fetch_names, rt_outputs)}
        self._cur_train_step += 1
        self._check_save()
        self._cur_train_epoch = (self._cur_train_step-1) // self._steps_pur_epoch
        return rt_outputs

    def predict_one_batch(self, batch):
        if gpu_dev_count > 1:
            feed, mask = batch
            rt_outputs = self._exe.run(self._distribute_pred_prog, feed=feed, fetch_list=self._pred_fetch_list)
            num_fakes = decode_fake(len(rt_outputs[0]), mask, self._predict_batch_size)
            if num_fakes:
                rt_outputs = [i[:-num_fakes] for i in rt_outputs]
        else:
            feed = self._pred_feed_batch_process_fn(batch)
            rt_outputs = self._exe.run(self._distribute_pred_prog, feed=feed, fetch_list=self._pred_fetch_list)

        rt_outputs = {k:v for k,v in zip(self._pred_fetch_name_list, rt_outputs)}
        return rt_outputs



    @property
    def name(self):
        return self._name
    
    @property
    def num_examples(self):
        return self._num_examples

    @property
    def mix_ratio(self):
        return self._mix_ratio

    @mix_ratio.setter
    def mix_ratio(self, value):
        self._mix_ratio = value

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def cur_train_step(self):
        return self._cur_train_step

    @property
    def cur_train_epoch(self):
        return self._cur_train_epoch

    @property
    def steps_pur_epoch(self):
        return self._steps_pur_epoch

    def _build_head(self, net_inputs, phase, scope=""):
        self._check_phase(phase)
        if phase == 'train':
            output_vars = self._task_head.build(net_inputs, scope_name=scope)
        if phase == 'predict':
            output_vars = self._pred_head.build(net_inputs, scope_name=scope)
        return output_vars
    
    def _save(self, save_path, suffix=None):
        # dirpath = save_path.rstrip('/').rstrip('\\') + suffix
        if suffix is not None:
            dirpath = os.path.join(save_path, suffix)
        else:
            dirpath = save_path
        self._pred_input_varname_list = [str(i) for i in self._pred_input_varname_list]

        prog = self._pred_prog.clone()
        fluid.io.save_inference_model(dirpath, self._pred_input_varname_list, self._pred_fetch_var_list, self._exe, prog)

        conf = {}
        for k, strv in self._save_protocol.items(): 
            d = None
            v = locals()
            exec('d={}'.format(strv), globals(), v)
            conf[k] = v['d']
        with open(os.path.join(dirpath, '__conf__'), 'w') as writer:
            writer.write(json.dumps(conf, indent=1))
        print(self._name + ': predict model saved at ' + dirpath)
        sys.stdout.flush()

    
    def _load(self, infer_model_path=None):
        if infer_model_path is None:
            infer_model_path = self._save_infermodel_path
        for k,v in json.load(open(os.path.join(infer_model_path, '__conf__'))).items(): 
            strv = self._save_protocol[k]
            exec('{}=v'.format(strv))
        pred_prog, self._pred_input_varname_list, self._pred_fetch_var_list = \
            fluid.io.load_inference_model(infer_model_path, self._exe)
        print(self._name+': inference model loaded from ' + infer_model_path)
        sys.stdout.flush()
        return pred_prog

