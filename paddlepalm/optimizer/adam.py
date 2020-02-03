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
from paddlepalm.optimizer.base_optimizer import Optimizer

class Adam(Optimizer):

    def __init__(self, loss_var, lr, lr_schedualer=None):

        Optimizer.__init__(self, loss_var, lr, lr_schedualer=None)

        self._loss = loss_var
        self._lr = lr
        self._lr_schedualer = lr_schedualer
    
    def _build(self, grad_clip=None):

        if self._lr_schedualer is not None:
            self._lr = self._lr_schedualer._build(self._lr)

        optimizer = fluid.optimizer.Adam(learning_rate=self._lr)

        if grad_clip is not None:
            clip_norm_thres = grad_clip
            # When using mixed precision training, scale the gradient clip threshold
            # by loss_scaling
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm_thres))

        _, param_grads = optimizer.minimize(self._loss)
        return param_grads

    def get_cur_learning_rate(self):
        return self._lr


