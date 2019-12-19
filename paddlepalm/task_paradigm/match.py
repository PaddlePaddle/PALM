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
    matching
    '''
    def __init__(self, config, phase='train', siamese=False, backbone_config=None):
        self._is_training = phase == 'train'
        self._hidden_size = backbone_config['hidden_size']
        self._batch_size = config['batch_size']
        self._is_siamese = siamese
        # self._label_pairwise = fluid.layers.fill_constant(shape=[self._batch_size, 1], value=1, dtype='float32')
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

    
    @property
    def inputs_attrs(self):
        if self._is_training and self._learning_strategy == 'pointwise':
            reader = {"label_ids": [[-1], 'int64']}
        else:
            reader = {}
        if self._learning_strategy == 'pairwise' and self._is_training:
            bb = {"sentence_pair_embedding": [[-1, self._hidden_size], 'float32'], "sentence_pair_embedding_neg": [[-1, self._hidden_size], 'float32']}
        else:
            bb = {"sentence_pair_embedding": [[-1, self._hidden_size], 'float32']}
            
        return {'reader': reader, 'backbone': bb}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {"loss": [[1], 'float32']}
        else:
            return {"logits": [[-1,2], 'float32']}

    def build(self, inputs, scope_name=""):
        
        if self._is_training and self._learning_strategy == 'pointwise':
            labels = inputs["reader"]["label_ids"] 
        cls_feats = inputs["backbone"]["sentence_pair_embedding"]
    
        if self._learning_strategy == 'pairwise' and self._is_training:
            cls_feats_neg = inputs["backbone"]["sentence_pair_embedding_neg"]

        if self._is_training:
            cls_feats = fluid.layers.dropout(
                x=cls_feats,
                dropout_prob=self._dropout_prob,
                dropout_implementation="upscale_in_train")
            if self._learning_strategy == 'pairwise':
                cls_feats_neg = fluid.layers.dropout(
                x=cls_feats_neg,
                dropout_prob=self._dropout_prob,
                dropout_implementation="upscale_in_train")
        
    
        # loss
        if self._learning_strategy == 'pairwise' and self._is_training:
            pos_score = fluid.layers.fc(
                input=cls_feats,
                size=1,
                act = "sigmoid",
                param_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_w",
                    initializer=self._param_initializer),
                bias_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_b",
                    initializer=fluid.initializer.Constant(0.)))
            neg_score = fluid.layers.fc(
                input=cls_feats_neg,
                size=1,
                act = "sigmoid",
                param_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_w",
                    initializer=self._param_initializer),
                bias_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_b",
                    initializer=fluid.initializer.Constant(0.)))        
           
            pos_score = fluid.layers.reshape(x=pos_score, shape=[-1, 1], inplace=True)
            neg_score = fluid.layers.reshape(x=neg_score, shape=[-1, 1], inplace=True)
        
            def computeHingeLoss(pos, neg):
                loss_part1 = fluid.layers.elementwise_sub(
                    fluid.layers.fill_constant_batch_size_like(
                        input=pos, shape=[-1, 1], value=self._margin, dtype='float32'), pos)
                loss_part2 = fluid.layers.elementwise_add(loss_part1, neg)
                loss_part3 = fluid.layers.elementwise_max(
                    fluid.layers.fill_constant_batch_size_like(
                        input=loss_part2, shape=[-1, 1], value=0.0, dtype='float32'), loss_part2)
                return loss_part3

            loss = fluid.layers.mean(computeHingeLoss(pos_score, neg_score))
            return {'loss': loss}
        
        else:
            logits = fluid.layers.fc(
                input=cls_feats,
                size=2,
                param_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_w",
                    initializer=self._param_initializer),
                bias_attr=fluid.ParamAttr(
                    name=scope_name+"cls_out_b",
                    initializer=fluid.initializer.Constant(0.)))
            
             
            if self._is_training:
                logits = fluid.layers.softmax(logits)
                ce_loss = fluid.layers.cross_entropy(
                    input=logits, label=labels)
                loss = fluid.layers.mean(x=ce_loss)
                return {'loss': loss}
            else:
                return {'logits': logits}

    def postprocess(self, rt_outputs):
        if not self._is_training:
            preds = rt_outputs['logits']
            preds = np.softmax(preds, -1)
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

                
