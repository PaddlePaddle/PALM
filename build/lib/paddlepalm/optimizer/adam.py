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
"""Optimization and learning rate scheduling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid

def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)

        return lr


def optimize(loss, config, max_train_steps=None, warmup_steps=0, train_program=None):
    if warmup_steps > 0:
        decay_strategy = config.get('lr_scheduler', 'linear_warmup_decay')
        if decay_strategy == 'noam_decay':
            scheduled_lr = fluid.layers.learning_rate_scheduler\
             .noam_decay(1/(warmup_steps *(config['learning_rate'] ** 2)),
                         warmup_steps)
        elif decay_strategy == 'linear_warmup_decay':
            scheduled_lr = linear_warmup_decay(config['learning_rate'], warmup_steps,
                                               max_train_steps)
        else:
            raise ValueError("Unkown lr_scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay'")
        optimizer = fluid.optimizer.Adam(learning_rate=scheduled_lr)
    else:
        optimizer = fluid.optimizer.Adam(learning_rate=config['learning_rate'])
        scheduled_lr = config['learning_rate']

    clip_norm_thres = 1.0
    # When using mixed precision training, scale the gradient clip threshold
    # by loss_scaling
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm_thres))

    def exclude_from_weight_decay(name):
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    param_list = dict()

    for param in train_program.global_block().all_parameters():
        param_list[param.name] = param * 1.0
        param_list[param.name].stop_gradient = True

    _, param_grads = optimizer.minimize(loss)


    if config.get('weight_decay', 0) > 0:
        for param, grad in param_grads:
            if exclude_from_weight_decay(param.name):
                continue
            with param.block.program._optimized_guard(
                [param, grad]), fluid.framework.name_scope("weight_decay"):
                updated_param = param - param_list[
                    param.name] * config['weight_decay'] * scheduled_lr
                fluid.layers.assign(output=param, input=updated_param)

