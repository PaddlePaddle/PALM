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
from paddle.fluid import layers
from paddlepalm.interface import task_paradigm
import numpy as np
import os

class TaskParadigm(task_paradigm):
    '''
    classification
    '''
    def __init__(self, config, phase, backbone_config=None):
        self._is_training = phase == 'train'
        self._hidden_size = backbone_config['hidden_size']
        self.num_classes = config['n_classes']
    
        if 'initializer_range' in config:
            self._param_initializer = config['initializer_range']
        else:
            self._param_initializer = fluid.initializer.TruncatedNormal(
                scale=backbone_config.get('initializer_range', 0.02))
        if 'dropout_prob' in config:
            self._dropout_prob = config['dropout_prob']
        else:
            self._dropout_prob = backbone_config.get('hidden_dropout_prob', 0.0)
        self._pred_output_path = config.get('pred_output_path', None)
        self._preds = []

    @property
    def inputs_attrs(self):
        if self._is_training:
            reader = {"label_ids": [[-1], 'int64']}
        else:
            reader = {}
        bb = {"sentence_embedding": [[-1, self._hidden_size], 'float32']}
        return {'reader': reader, 'backbone': bb}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {'loss': [[1], 'float32']}
        else:
            return {'logits': [[-1, self.num_classes], 'float32']}

    def build(self, inputs, scope_name=''):
        sent_emb = inputs['backbone']['sentence_embedding']
        if self._is_training:
            label_ids = inputs['reader']['label_ids']
            cls_feats = fluid.layers.dropout(
                x=sent_emb,
                dropout_prob=self._dropout_prob,
                dropout_implementation="upscale_in_train")

        logits = fluid.layers.fc(
            input=sent_emb,
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name=scope_name+"cls_out_w",
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(
                name=scope_name+"cls_out_b", initializer=fluid.initializer.Constant(0.)))

        if self._is_training:
            inputs = fluid.layers.softmax(logits)
            loss = fluid.layers.cross_entropy(
                input=inputs, label=label_ids)
            loss = layers.mean(loss)
            return {"loss": loss}
        else:
            return {"logits":logits}

    def postprocess(self, rt_outputs):
        if not self._is_training:
            logits = rt_outputs['logits']
            preds = np.argmax(logits, -1)
            self._preds.extend(preds.tolist())

    def epoch_postprocess(self, post_inputs):
        # there is no post_inputs needed and not declared in epoch_inputs_attrs, hence no elements exist in post_inputs
        if not self._is_training:
            if self._pred_output_path is None:
                raise ValueError('argument pred_output_path not found in config. Please add it into config dict/file.')
            with open(os.path.join(self._pred_output_path, 'predictions.json'), 'w') as writer:
                for p in self._preds:
                    writer.write(str(p)+'\n')
            print('Predictions saved at '+os.path.join(self._pred_output_path, 'predictions.json'))

                
