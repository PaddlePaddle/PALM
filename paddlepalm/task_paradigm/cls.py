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

import paddle.fluid as fluid
from paddlepalm.interface import task_paradigm
from paddle.fluid import layers

class TaskParadigm(task_paradigm):
    '''
    classification
    '''
    def __init___(self, config, phase):
        self._is_training = phase == 'train'
        self.sent_emb_size = config['hidden_size']
        self.num_classes = config['n_classes']
    
    @property
    def inputs_attrs(self):
        return {'bakcbone': {"sentence_emb": [-1, self.sent_emb_size], 'float32']},
                'reader': {"label_ids": [[-1, 1], 'int64']}}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {'loss': [[1], 'float32']}
        else:
            return {'logits': [-1, self.num_classes], 'float32'}

    def build(self, **inputs):
        sent_emb = inputs['backbone']['sentence_emb']
        label_ids = inputs['reader']['label_ids']

        logits = fluid.layers.fc(
            input=ent_emb
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.1)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

        loss = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=label_ids)
        loss = layers.mean(loss)
        if self._is_training:
            return {"loss": loss}
        else:
            return {"logits":logits}
