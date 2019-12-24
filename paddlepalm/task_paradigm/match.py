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
import json


def computeHingeLoss(pos, neg, margin):
    loss_part1 = fluid.layers.elementwise_sub(
        fluid.layers.fill_constant_batch_size_like(
            input=pos, shape=[-1, 1], value=margin, dtype='float32'), pos)
    loss_part2 = fluid.layers.elementwise_add(loss_part1, neg)
    loss_part3 = fluid.layers.elementwise_max(
        fluid.layers.fill_constant_batch_size_like(
            input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'), loss_part2)
    return loss_part3


class TaskParadigm(task_paradigm):
    '''
    matching
    '''
    def __init__(self, config, phase, backbone_config=None):
        self._is_training = phase == 'train'
        self._hidden_size = backbone_config['hidden_size']
        self._batch_size = config['batch_size']

        self._num_classes = config.get('num_classes', 2)
        if 'learning_strategy' in config:
            self._learning_strategy = config['learning_strategy']
        else:
            self._learning_strategy = 'pointwise'
        
        if 'margin' in config:
            self._margin = config['margin']
        else:
            self._margin = 0.5

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
        self._preds_logits = []

    
    @property
    def inputs_attrs(self):
        reader = {}
        bb = {"sentence_pair_embedding": [[-1, self._hidden_size], 'float32']}
        if self._is_training:
            if self._learning_strategy == 'pointwise':
                reader["label_ids"] = [[-1], 'int64']
            elif self._learning_strategy == 'pairwise':
                bb["sentence_pair_embedding_neg"] = [[-1, self._hidden_size], 'float32']

        return {'reader': reader, 'backbone': bb}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {"loss": [[1], 'float32']}
        else:
            if self._learning_strategy=='paiwise':
                return {"probs": [[-1, 1], 'float32']}
            else:
                return {"logits": [[-1, 2], 'float32'],
                        "probs": [[-1, 2], 'float32']}

    def build(self, inputs, scope_name=""):

        # inputs          
        cls_feats = inputs["backbone"]["sentence_pair_embedding"] 
        if self._is_training:
            cls_feats = fluid.layers.dropout(
                x=cls_feats,
                dropout_prob=self._dropout_prob,
                dropout_implementation="upscale_in_train")
            if self._learning_strategy == 'pairwise':
                cls_feats_neg = inputs["backbone"]["sentence_pair_embedding_neg"]
                cls_feats_neg = fluid.layers.dropout(
                x=cls_feats_neg,
                dropout_prob=self._dropout_prob,
                dropout_implementation="upscale_in_train")
            elif self._learning_strategy == 'pointwise':
                labels = inputs["reader"]["label_ids"] 
        
        # loss
        # for pointwise
        if self._learning_strategy == 'pointwise':
            logits = fluid.layers.fc(
                input=cls_feats,
                size=self._num_classes,
                param_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_w",
                    initializer=self._param_initializer),
                bias_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_b",
                    initializer=fluid.initializer.Constant(0.)))
            probs = fluid.layers.softmax(logits)
            if self._is_training:
                ce_loss = fluid.layers.cross_entropy(
                    input=probs, label=labels)
                loss = fluid.layers.mean(x=ce_loss)
                return {'loss': loss}
            # for pred
            else:
                return {'logits': logits,
                        'probs': probs}
        # for pairwise
        elif self._learning_strategy == 'pairwise':
            pos_score = fluid.layers.fc(
                input=cls_feats,
                size=1,
                act = "sigmoid",
                param_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_w_pr",
                    initializer=self._param_initializer),
                bias_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_b_pr",
                    initializer=fluid.initializer.Constant(0.)))
            pos_score = fluid.layers.reshape(x=pos_score, shape=[-1, 1], inplace=True)

            if self._is_training:
                neg_score = fluid.layers.fc(
                    input=cls_feats_neg,
                    size=1,
                    act = "sigmoid",
                    param_attr=fluid.ParamAttr(
                        name=scope_name+"cls_out_w_pr",
                        initializer=self._param_initializer),
                    bias_attr=fluid.ParamAttr(
                        name=scope_name+"cls_out_b_pr",
                        initializer=fluid.initializer.Constant(0.)))        
                neg_score = fluid.layers.reshape(x=neg_score, shape=[-1, 1], inplace=True)
        
                loss = fluid.layers.mean(computeHingeLoss(pos_score, neg_score, self._margin))
                return {'loss': loss}
            # for pred
            else:
                return {'probs': pos_score}
        


    def postprocess(self, rt_outputs):
        if not self._is_training:
            probs = []
            logits = []
            probs = rt_outputs['probs']
            self._preds.extend(probs.tolist())
            if self._learning_strategy == 'pointwise':
                logits = rt_outputs['logits']
                self._preds_logits.extend(logits.tolist())
        
    def epoch_postprocess(self, post_inputs):
        # there is no post_inputs needed and not declared in epoch_inputs_attrs, hence no elements exist in post_inputs
        if not self._is_training:
            if self._pred_output_path is None:
                raise ValueError('argument pred_output_path not found in config. Please add it into config dict/file.')
            with open(os.path.join(self._pred_output_path, 'predictions.json'), 'w') as writer:
                for i in range(len(self._preds)):
                    if self._learning_strategy == 'pointwise':
                        label = 0 if self._preds[i][0] > self._preds[i][1] else 1
                        result = {'index': i, 'label': label, 'logits': self._preds_logits[i], 'probs': self._preds[i]}
                    elif self._learning_strategy == 'pairwise':
                        label = 0 if self._preds[i][0] < 0.5 else 1
                        result = {'index': i, 'label': label, 'probs': self._preds[i][0]}
                    result = json.dumps(result)
                    writer.write(result+'\n')
            print('Predictions saved at '+os.path.join(self._pred_output_path, 'predictions.json'))