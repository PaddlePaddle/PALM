# -*- coding: UTF-8 -*-
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

from paddlepalm.interface import reader as base_reader
from paddlepalm.interface import task_paradigm as base_paradigm
import os
import json
from paddle import fluid
import importlib
from paddlepalm.default_settings import *


def check_req_args(conf, name):
    assert 'reader' in conf, name+': reader is required to build TaskInstance.'
    assert 'paradigm' in conf, name+': paradigm is required to build TaskInstance.'
    assert 'train_file' in conf or 'pred_file' in conf, name+': at least train_file or pred_file should be provided to build TaskInstance.'


class TaskInstance(object):
    
    def __init__(self, name, id, config, verbose=True):
        self._name = name
        self._config = config
        self._verbose = verbose
        self._id = id

        check_req_args(config, name)

        # parse Reader and Paradigm
        self.reader_name = config['reader']
        reader_mod = importlib.import_module(READER_DIR + '.' + self.reader_name)
        Reader = getattr(reader_mod, 'Reader')

        parad_name = config['paradigm']
        parad_mod = importlib.import_module(PARADIGM_DIR + '.' + parad_name)
        Paradigm = getattr(parad_mod, 'TaskParadigm')

        self._Reader = Reader
        self._Paradigm = Paradigm

        self._save_infermodel_path = os.path.join(self._config['save_path'], self._name, 'infer_model')
        self._save_ckpt_path = os.path.join(self._config['save_path'], 'ckpt')
        self._save_infermodel_every_n_steps = config.get('save_infermodel_every_n_steps', -1)

        # following flags can be fetch from instance config file
        self._is_target = config.get('is_target', True)
        self._first_target = config.get('is_first_target', False)
        self._task_reuse_scope = config.get('task_reuse_scope', name)

        self._feeded_var_names = None
        self._target_vars = None

        # training process management
        self._mix_ratio = None
        self._expected_train_steps = None
        self._expected_train_epochs = None
        self._steps_pur_epoch = None
        self._cur_train_epoch = 0
        self._cur_train_step = 0
        self._train_finish = False

        # 存放不同运行阶段（train，eval，pred）的数据集reader，key为phase，value为Reader实例
        self._reader = {'train': None, 'eval': None, 'pred': None}
        self._input_layer = None
        self._inputname_to_varname = {}
        self._task_layer = {'train': None, 'eval': None, 'pred': None}
        self._pred_input_name_list = []
        self._pred_input_varname_list = []
        self._pred_fetch_name_list = []
        self._pred_fetch_var_list = []

        self._exe = fluid.Executor(fluid.CPUPlace())

        self._save_protocol = {
            'input_names': 'self._pred_input_name_list',
            'input_varnames': 'self._pred_input_varname_list',
            'fetch_list': 'self._pred_fetch_name_list'}


    def build_task_layer(self, net_inputs, phase, scope=""):
        output_vars = self._task_layer[phase].build(net_inputs, scope_name=scope)
        if phase == 'pred':
            if output_vars is not None:
                self._pred_fetch_name_list, self._pred_fetch_var_list = zip(*output_vars.items())
            else:
                self._pred_fetch_name_list = []
                self._pred_fetch_var_list = []
        return output_vars

    def postprocess(self, rt_outputs, phase):
        return self._task_layer[phase].postprocess(rt_outputs)

    def epoch_postprocess(self, epoch_inputs, phase):
        return self._task_layer[phase].epoch_postprocess(epoch_inputs)
    
    def save(self, suffix='', prog=None):
        dirpath = self._save_infermodel_path + suffix
        self._pred_input_varname_list = [str(i) for i in self._pred_input_varname_list]

        # fluid.io.save_inference_model(dirpath, self._pred_input_varname_list, self._pred_fetch_var_list, self._exe, export_for_deployment = True)
        # prog = fluid.default_main_program().clone()
        if prog is not None:
            save_prog = prog
        else:
            save_prog = fluid.default_main_program().clone()

        fluid.io.save_inference_model(dirpath, self._pred_input_varname_list, self._pred_fetch_var_list, self._exe, save_prog)

        conf = {}
        for k, strv in self._save_protocol.items(): 
            d = None
            v = locals()
            exec('d={}'.format(strv), globals(), v)
            conf[k] = v['d']
        with open(os.path.join(dirpath, '__conf__'), 'w') as writer:
            writer.write(json.dumps(conf, indent=1))
        print(self._name + ': inference model saved at ' + dirpath)

    def load(self, infer_model_path=None):
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
    def tid(self):
        return self._id

    @property
    def Reader(self):
        return self._Reader

    # @Reader.setter
    # def Reader(self, cls):
    #     assert base_reader.__name__ == cls.__bases__[-1].__name__, \
    #         "expect: {}, receive: {}.".format(base_reader.__name__, \
    #                                           cls.__bases__[-1].__name__)
    #     self._Reader = cls

    @property
    def Paradigm(self):
        return self._Paradigm

    # @Paradigm.setter
    # def Paradigm(self, cls):
    #     assert base_paradigm.__name__ == cls.__bases__[-1].__name__, \
    #         "expect: {}, receive: {}.".format(base_paradigm.__name__, \
    #                                           cls.__bases__[-1].__name__)
    #     self._Paradigm = cls

    @property
    def config(self):
        return self._config

    @property
    def reader(self):
        return self._reader

    @property
    def pred_input(self):
        return dict(zip(*[self._pred_input_name_list, self._pred_input_varname_list]))

    @pred_input.setter
    def pred_input(self, val):
        assert isinstance(val, dict)
        self._pred_input_name_list, self._pred_input_varname_list = \
            zip(*[[k, v.name] for k,v in val.items()])

    @property
    def pred_fetch_list(self):
        return [self._pred_fetch_name_list, self._pred_fetch_var_list]

    @property
    def task_layer(self):
        return self._task_layer

    @property
    def is_first_target(self):
        return self._is_first_target

    @is_first_target.setter
    def is_first_target(self, value):
        self._is_first_target = bool(value)
        if self._is_first_target:
            assert self._is_target, "ERROR: only target task could be set as main task."
        if self._verbose and self._is_first_target:
            print("{}: set as main task".format(self._name))

    @property
    def is_target(self):
        if self._is_target is not None:
            return self._is_target
        else:
            raise ValueError("{}: is_target is None".format(self._name))

    @is_target.setter
    def is_target(self, value):
        self._is_target = bool(value)
        if self._verbose:
            if self._is_target:
                print('{}: set as target task.'.format(self._name))
            else:
                print('{}: set as aux task.'.format(self._name))

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

    @cur_train_epoch.setter
    def cur_train_epoch(self, value):
        self._cur_train_epoch = value

    @property
    def cur_train_step(self):
        return self._cur_train_step

    @cur_train_step.setter
    def cur_train_step(self, value):
        self._cur_train_step = value
        if self._cur_train_step > self._steps_pur_epoch:
            self._cur_train_epoch += 1
            self._cur_train_step = 1
        if self._is_target and self._cur_train_step + self._cur_train_epoch * self._steps_pur_epoch >= self._expected_train_steps:
            self._train_finish = True

    @property
    def steps_pur_epoch(self):
        return self._steps_pur_epoch

    @steps_pur_epoch.setter
    def steps_pur_epoch(self, value):
        self._steps_pur_epoch = value

    @property
    def train_finish(self):
        return self._train_finish

    @property
    def task_reuse_scope(self):
        if self._task_reuse_scope is not None:
            return self._task_reuse_scope
        else:
            raise ValueError("{}: task_reuse_scope is None".format(self._name))

    @task_reuse_scope.setter
    def task_reuse_scope(self, scope_name):
        self._task_reuse_scope = str(scope_name)
        if self._verbose:
            print('{}: task_reuse_scope is set to {}'.format(self._name, self._task_reuse_scope))





        

def check_instances(insts):
    """to check ids, first_target"""
    pass

def _check_ids():
    pass

def _check_targets():
    pass

def _check_reuse_scopes():
    pass
