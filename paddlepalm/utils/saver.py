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

from __future__ import print_function

import os
import six
import ast
import copy
import tarfile
import shutil

import numpy as np
import paddle.fluid as fluid

def init_checkpoint(exe, init_checkpoint_path, main_program, skip_list = []):
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

    def existed_persitables(var):
        if not fluid.io.is_persistable(var):
            return False
        if var.name in skip_list:
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print("Load model from {}".format(init_checkpoint_path))


def init_pretraining_params(exe,
                            pretraining_params_path,
                            convert,
                            main_program,
                            strict=False):
                            
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    if convert:
        assert os.path.exists(os.path.join(pretraining_params_path, '__palmmodel__')), "__palmmodel__ not found."

        with tarfile.open(os.path.join(pretraining_params_path, '__palmmodel__'), 'r') as f:
            f.extractall(os.path.join(pretraining_params_path, '.temp'))
        
        log_path = os.path.join(pretraining_params_path, '__palmmodel__')
        pretraining_params_path = os.path.join(pretraining_params_path, '.temp')

    else:
        log_path = pretraining_params_path
    
    print("Loading pretraining parameters from {}...".format(pretraining_params_path))

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        if not os.path.exists(os.path.join(pretraining_params_path, var.name)):
            if strict:
                raise Exception('Error: {} not found in {}.'.format(var.name, log_path))
            else:
                print('Warning: {} not found in {}.'.format(var.name, log_path))
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    if convert:
        shutil.rmtree(pretraining_params_path)
    print('')


