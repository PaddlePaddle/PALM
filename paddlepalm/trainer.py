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
import numpy as np
import paddlepalm.utils.basic_helper as helper
from paddlepalm.utils import reader_helper, saver
from paddlepalm.distribute import gpu_dev_count, data_feeder
# from paddlepalm.default_settings import *

DEBUG=False


class Trainer(object):

    def __init__(self, name, reader, task_head, \
                 mix_ratio=1.0, reuse_head_with=None, \
                 silent=False):

        self._name = name
        self._verbose = not silent
        self._reader = reader
        self._pred_reader = None
        self._task_head = task_head
        self._pred_head = pred_head

        # if save_predict_model:
        #     self._save_predict_model = True
        #     assert pred_head is not None, "pred_head is required to save predict model."
        #     self._pred_reader = reader.clone(phase='pred')
        # else:
        #     assert pred_head is None, "You should set save_predict_model as True, or the pred_head is invalid." 
        #     self._save_predict_model = False
        #     self._pred_reader = None

        # self._save_steps = save_steps

        self._task_reuse_scope = name if reuse_head_with is None else reuse_head_with

        self._feeded_var_names = None
        self._target_vars = None

        self._num_examples = 0

        # training process management
        self._mix_ratio = mix_ratio
        self._expected_train_steps = None
        self._expected_train_epochs = None
        self._steps_pur_epoch = None
        self._cur_train_epoch = 0
        self._cur_train_step = 0
        self._train_finish = False

        # 存放不同运行阶段（train，eval，pred）的数据集reader，key为phase，value为Reader实例
        # self._reader = {'train': reader, 'eval': None, 'pred': self._pred_reader}
        # self._input_layer = None
        self._inputname_to_varname = {}
        # self._task_layer = {'train': task_head, 'eval': None, 'pred': pred_head}
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
        self._build_forward = False

    def build_predict_head(self, pred_backbone, pred_prog=None, pred_init_prog=None):
        pred_task_attr_from_reader = helper.encode_inputs(self._pred_head.inputs_attrs['reader'], self.name)
        # pred_task_attr_from_reader = self._pred_head.inputs_attrs['reader']

        # _check_io(pred_backbone.inputs_attr, pred_reader.outputs_attr, in_name=bb_name+'_backbone', out_name='reader.pred')
        # _check_io(pred_parad.inputs_attrs['reader'], pred_reader.outputs_attr, in_name='task_paradigm.pred.reader', out_name='reader.pred')
        # _check_io(pred_parad.inputs_attrs['backbone'], pred_backbone.outputs_attr, in_name='task_paradigm.pred.backbone', out_name=bb_name+'_backbone')
        pred_input_names, pred_shape_and_dtypes, _ = reader_helper.merge_input_attrs(backbone.inputs_attr, pred_task_attr_from_reader, insert_taskid=False, insert_batchsize=False, insert_seqlen=False, insert_batchsize_x_seqlen=False)
        pred_input_attrs = [[i, j, k] for i, (j,k) in zip(pred_input_names, pred_shape_and_dtypes)]
        
        if pred_prog is None:
            pred_prog = fluid.Program()
        if pred_init_prog is None:
            pred_init_prog = fluid.Program()
        with fluid.program_guard(pred_prog, pred_init_prog):
            pred_net_inputs = reader_helper.create_net_inputs(pred_input_attrs)
            # pred_bb_output_vars = pred_backbone.build(pred_net_inputs, scope_name='__paddlepalm_')
            pred_bb_output_vars = pred_backbone.build(pred_net_inputs)

        # prepare predict vars for saving inference model
        with fluid.program_guard(pred_prog, pred_init_prog):
            cur_inputs = helper.decode_inputs(pred_net_inputs, self.name)
            # self.pred_input = cur_inputs
            self._pred_input_name_list, self._pred_input_varname_list = \
                zip(*[[k, v.name] for k,v in cur_inputs.items()])

            pred_task_inputs = {'backbone': pred_bb_output_vars, 'reader': cur_inputs}
            scope = self.name + '.'
            with fluid.unique_name.guard(scope):
                self._build_head(pred_task_inputs, phase='pred', scope=scope)




    def build_forward(self, backbone, pred_backbone=None, train_prog=None, train_init_prog=None, pred_prog=None, pred_init_prog=None):

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
        input_names, shape_and_dtypes, name_to_position = reader_helper.merge_input_attrs(backbone.inputs_attr, task_attr_from_reader, insert_taskid=False, insert_batchsize=False, insert_seqlen=False, insert_batchsize_x_seqlen=False)
        # shapes: [task_id, shapes_of_backbone, shapes_of_inst1, ..., shapes_of_instN]
        self._shape_and_dtypes = shape_and_dtypes
        self._name_to_position = name_to_position

        if DEBUG:
            print('----- for debug -----')
            print('joint input names:')
            print(joint_input_names)
            print('joint input shape and dtypes:')
            print(joint_shape_and_dtypes)


        input_attrs = [[i, j, k] for i, (j,k) in zip(input_names, shape_and_dtypes)]

        if train_prog is None:
            train_prog = fluid.Program()
        if train_init_prog is None:
            train_init_prog = fluid.Program()
        self._prog = train_prog
        self._train_prog = train_prog
        self._train_init_prog = train_init_prog
        with fluid.program_guard(train_prog, train_init_prog):
            net_inputs = reader_helper.create_net_inputs(input_attrs, async=False)
            self._net_inputs = net_inputs

            # build backbone and task layers
            # bb_output_vars = self._backbone.build(net_inputs, scope_name='__paddlepalm_')
            bb_output_vars = backbone.build(net_inputs)
            assert sorted(bb_output_vars.keys()) == sorted(backbone.outputs_attr.keys())
        

        # fluid.framework.switch_main_program(train_prog)
        # fluid.framework.switch_startup_program(train_init_prog)

        task_output_vars = {}
        task_inputs = {'backbone': bb_output_vars}
        task_inputs_from_reader = helper.decode_inputs(net_inputs, self.name)
        task_inputs['reader'] = task_inputs_from_reader

        scope = self.name+'.'
        with fluid.program_guard(train_prog, train_init_prog):
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
        with fluid.program_guard(train_prog, train_init_prog):
            loss_var = fluid.layers.reduce_sum(task_output_vars[self.name+'.loss'])

        self._distribute_train_prog = fluid.CompiledProgram(self._train_prog).with_data_parallel(loss_name=loss_var.name)
        return loss_var

    def build_backward(self, optimizer, weight_decay=None, use_ema=False, ema_decay=0.9999):
        # build optimizer
        optimizer._set_prog(self._train_prog)
        with fluid.program_guard(self._train_prog, self._train_init_prog):
            param_grads = optimizer.build()

            if weight_decay is not None:

                param_list = dict()

                for param in self._prog.global_block().all_parameters():
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

    def load_data(self, input_file, file_format, batch_size, num_epochs=None, shuffle_train=True):
        # load data
        print("preparing data...", end='')
        self._reader._load_data(input_file=input_file, batch_size=batch_size, \
                                num_epochs=num_epochs, file_format=file_format, \
                                shuffle_train=shuffle_train)
        self._num_examples = self._reader.num_examples
        # 这里不确定是否要向上取整，需确认
        # tail = self._num_examples % batch_size > 0
        # self._steps_pur_epoch = self._num_examples // batch_size + 1 if tail else 0
        self._steps_pur_epoch = self._num_examples // batch_size
        print('ok!')

        # merge dataset iterators and create net input vars
        iterator = self._reader._iterator()
        prefix = self.name

        # 对yield出的数据进行runtime检查和适配
        iterator_fn = reader_helper.create_iterator_fn(iterator, prefix, self._shape_and_dtypes, self._name_to_position, return_type='dict')
        feed_batch_process_fn = reader_helper.create_feed_batch_process_fn(self._net_inputs)
        self._feed_batch_process_fn = feed_batch_process_fn
        if gpu_dev_count > 1:
            distribute_feeder_fn = data_feeder(iterator_fn, feed_batch_process_fn)
        else:
            distribute_feeder_fn = iterator_fn
        return distribute_feeder_fn()

    def random_init_params(self):
        on_gpu = gpu_dev_count > 0
        self._exe = helper.build_executor(on_gpu)
        print('random init params...')
        self._exe.run(self._train_init_prog)

    def load_pretrain(self, model_path):
        # load pretrain model (or ckpt)
        assert self._exe is not None, "You need to random_init_params before load pretrain models."

        saver.init_pretraining_params(
            self._exe,
            model_path,
            main_program=self._train_init_prog)

    def set_predict_head(self):
        pass

    def train(self, iterator, save_path=None, save_steps=None, save_type='ckpt', print_steps=5):

        save_type = save_type.split(',')
        if 'predict' in save_type:
            assert self._pred_head is not None, "Predict head not found! You should call set_predict_head first if you want to save predict model."
            assert save_path is not None and save_steps is not None, 'save_path and save_steps is required to save model.'
            save_predict = True
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_predict = False

        if 'ckpt' in save_type:
            if save_path is not None and save_steps is not None:
                save_ckpt = True
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            else:
                "WARNING: save_path or save_steps is not set, model will not be saved during training."
                save_ckpt = False
        else:
            save_ckpt = False

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
            # rt_outputs = self.exe.run(self._train_prog, feed=feed, fetch_list=self._fetch_list)
            # print(rt_outputs)
            # print(len(rt_outputs))
            # if gpu_dev_count > 1:
            #     while mask.pop() == False:
            #         rt_outputs.pop()

            # rt_outputs = {k:v for k,v in zip(self._fetch_names, rt_outputs)}

            task_rt_outputs = {k[len(self.name+'.'):]: v for k,v in rt_outputs.items() if k.startswith(self.name+'.')}
            self._task_head.postprocess(task_rt_outputs)

            self._cur_train_step += 1
            self._cur_train_epoch = (self._cur_train_step-1) // self._steps_pur_epoch

            # if self._save_predict_model and self._cur_train_step % save_steps == 0:
            #     self.save(save_path, suffix='.step'+str(self._cur_train_steps))

            if print_steps > 0 and self._cur_train_step % print_steps == 0:
                loss = rt_outputs[self.name+'.loss']
                loss = np.mean(np.squeeze(loss)).tolist()

                time_end = time.time()
                time_cost = time_end - time_begin

                print("step {}/{} (epoch {}), loss: {:.3f}, speed: {:.2f} steps/s".format(
                       (self._cur_train_step-1) % self._steps_pur_epoch + 1, self._steps_pur_epoch, self._cur_train_epoch,
                       loss, print_steps / time_cost))
                time_begin = time.time()

            # if cur_task.train_finish and cur_task.cur_train_step + cur_task.cur_train_epoch * cur_task.steps_pur_epoch == cur_task.expected_train_steps:
            #     print(cur_task.name+': train finished!')
            #     cur_task.save()

            if (save_predict or save_ckpt) and self._cur_train_step % save_steps == 0:
                if save_predict_model:
                    self.save(save_path, suffix='pred.step'+str(global_step))
                if save_ckpt:
                    fluid.io.save_persistables(self.exe, os.path.join(save_path, 'ckpt.step'+str(global_step)), self._train_prog)
                    print('checkpoint has been saved at '+os.path.join(save_path, 'ckpt.step'+str(global_step)))

        # save_path = os.path.join(main_conf['save_path'], 'ckpt',
        #                          "step_" + str(global_step))
        # fluid.io.save_persistables(self.exe, save_path, saver_program)
        # print('checkpoint has been saved at '+save_path)

        # print("ALL tasks train finished, exiting...")

    def train_one_step(self, batch):
        if gpu_dev_count > 1:
            feed, mask = batch
            rt_outputs = self.exe.run(self._distribute_train_prog, feed=feed, fetch_list=self._fetch_list)
            while mask.pop() == False:
                rt_outputs.pop()
        else:
            feed = self._feed_batch_process_fn(batch)
            rt_outputs = self._exe.run(self._distribute_train_prog, feed=feed, fetch_list=self._fetch_list)

        rt_outputs = {k:v for k,v in zip(self._fetch_names, rt_outputs)}
        return rt_outputs
        

    def _build_head(self, net_inputs, phase, scope=""):
        if phase == 'train':
            output_vars = self._task_head.build(net_inputs, scope_name=scope)
        if phase == 'pred':
            output_vars = self._pred_head.build(net_inputs, scope_name=scope)
            if output_vars is not None:
                self._pred_fetch_name_list, self._pred_fetch_var_list = zip(*output_vars.items())
            else:
                self._pred_fetch_name_list = []
                self._pred_fetch_var_list = []
        return output_vars

    def _postprocess(self, rt_outputs, phase):
        return self._task_layer[phase].postprocess(rt_outputs)

    def _epoch_postprocess(self, epoch_inputs, phase):
        return self._task_layer[phase].epoch_postprocess(epoch_inputs)
    
    def save(self, save_path, suffix=None):
        # dirpath = save_path.rstrip('/').rstrip('\\') + suffix
        if suffix is not None:
            dirpath = os.path.join(save_path, suffix)
        else:
            dirpath = save_path
        self._pred_input_varname_list = [str(i) for i in self._pred_input_varname_list]

        prog = fluid.default_main_program().clone()
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

    def _load(self, infer_model_path=None):
        if infer_model_path is None:
            infer_model_path = self._save_infermodel_path
        for k,v in json.load(open(os.path.join(infer_model_path, '__conf__'))).items(): 
            strv = self._save_protocol[k]
            exec('{}=v'.format(strv))
        pred_prog, self._pred_input_varname_list, self._pred_fetch_var_list = \
            fluid.io.load_inference_model(infer_model_path, self._exe)
        print(self._name+': inference model loaded from ' + infer_model_path)
        return pred_prog

    @property
    def name(self):
        return self._name

    @property
    def num_examples(self):
        return self._num_examples

    # @property
    # def _pred_input(self):
    #     return zip(*[self._pred_input_name_list, self._pred_input_varname_list])

    # @_pred_input.setter
    # def _pred_input(self, val):
    #     assert isinstance(val, dict)
    #     self._pred_input_name_list, self._pred_input_varname_list = \
    #         zip(*[[k, v.name] for k,v in val.items()])

    # @property
    # def _pred_fetch_list(self):
    #     return [self._pred_fetch_name_list, self._pred_fetch_var_list]

    @property
    def mix_ratio(self):
        if self._mix_ratio is not None:
            return self._mix_ratio
        else:
            raise ValueError("{}: mix_ratio is None".format(self._name))

    @mix_ratio.setter
    def mix_ratio(self, value):
        self._mix_ratio = float(value)
        if self._verbose:
            print('{}: mix_ratio is set to {}'.format(self._name, self._mix_ratio))

    @property
    def save_infermodel_every_n_steps(self):
        return self._save_infermodel_every_n_steps

    @save_infermodel_every_n_steps.setter
    def save_infermodel_every_n_steps(self, val):
        self._save_infermodel_every_n_steps = val

    @property
    def expected_train_steps(self):
        return self._expected_train_steps

    @expected_train_steps.setter
    def expected_train_steps(self, value):
        self._expected_train_steps = value
        self._expected_train_epochs = value / float(self._steps_pur_epoch)

    @property
    def expected_train_epochs(self):
        return self._expected_train_epochs

    @property
    def cur_train_epoch(self):
        return self._cur_train_epoch

    @property
    def cur_train_step(self):
        return self._cur_train_step

    # @cur_train_step.setter
    # def _cur_train_step(self, value):
    #     self._cur_train_step = value
    #     if self._cur_train_step > self._steps_pur_epoch:
    #         self._cur_train_epoch += 1
    #         self._cur_train_step = 1
    #     if self._is_target and self._cur_train_step + self._cur_train_epoch * self._steps_pur_epoch >= self._expected_train_steps:
    #         self._train_finish = True

    @property
    def steps_pur_epoch(self):
        return self._steps_pur_epoch

    @steps_pur_epoch.setter
    def steps_pur_epoch(self, value):
        self._steps_pur_epoch = value

    @property
    def train_finish(self):
        return self._train_finish

    def tasklayer_reuse_with(self, task):
        assert isinstance(task, Task)
        if self._lock:
            raise Exception('you can only set tasklayer reuses BEFORE Controller created.')
        self._task_reuse_scope = task.name
    
    def _set_lock(self):
        self._lock = True

