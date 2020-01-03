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
import math

class TaskParadigm(task_paradigm):
    '''
    Sequence labeling
    '''
    def __init__(self, config, phase, backbone_config=None):
        self._is_training = phase == 'train'
        self._hidden_size = backbone_config['hidden_size']
        self.num_classes = config['n_classes']
        self.learning_rate = config['learning_rate']
        
    
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
        self._use_crf = config.get('use_crf', False)
        self._preds = []


    @property
    def inputs_attrs(self):
        reader = {}
        bb = {"encoder_outputs": [[-1, -1, -1], 'float32']}
        if self._use_crf:
            reader["seq_lens"] = [[-1], 'int64']
        if self._is_training:
            reader["label_ids"] = [[-1, -1], 'int64']
        return {'reader': reader, 'backbone': bb}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {'loss': [[1], 'float32']}
        else:
            if self._use_crf:
                return {'crf_decode': [[-1, -1], 'float32']}
            else:
                return {'logits': [[-1, -1, self.num_classes], 'float32']}

    def build(self, inputs, scope_name=''):
        token_emb = inputs['backbone']['encoder_outputs']
        seq_lens = inputs['reader']['seq_lens']
        if self._is_training:
            label_ids = inputs['reader']['label_ids']

        logits = fluid.layers.fc(
            size=self.num_classes,
            input=token_emb,
            param_attr=fluid.ParamAttr(
                initializer=self._param_initializer,
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)),
            bias_attr=fluid.ParamAttr(
                name=scope_name+"cls_out_b", initializer=fluid.initializer.Constant(0.)),
            num_flatten_dims=2)

        # use_crf
        if self._use_crf:
            if self._is_training:
                crf_cost = fluid.layers.linear_chain_crf(  
                    input=logits,
                    label=label_ids,
                    param_attr=fluid.ParamAttr(
                        name=scope_name+'crfw', learning_rate=self.learning_rate),
                    length=seq_lens)
                
                avg_cost = fluid.layers.mean(x=crf_cost)
                crf_decode = fluid.layers.crf_decoding(
                    input=logits,
                    param_attr=fluid.ParamAttr(name=scope_name+'crfw'),
                    length=seq_lens)
                return {"loss": avg_cost}   
            else:
                size = self.num_classes
                fluid.layers.create_parameter(
                    shape=[size+2, size], dtype=logits.dtype, name=scope_name+'crfw')
                crf_decode = fluid.layers.crf_decoding(
                    input=logits, param_attr=fluid.ParamAttr(name=scope_name+'crfw'),
                    length=seq_lens)
                return {"crf_decode": crf_decode}    

        else:
            if self._is_training:
                probs = fluid.layers.softmax(logits)
                ce_loss = fluid.layers.cross_entropy(
                    input=probs, label=label_ids)
                avg_cost = fluid.layers.mean(x=ce_loss)
                return {"loss": avg_cost}
            else:
                return {"logits": logits}



    def postprocess(self, rt_outputs):
        if not self._is_training:
            if self._use_crf:
                preds = rt_outputs['crf_decode']
            else:
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
