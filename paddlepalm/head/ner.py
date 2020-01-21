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
from paddlepalm.head.base_head import Head
import numpy as np
import os
import math

class SequenceLabel(Head):
    '''
    Sequence label
    '''
    def __init__(self, num_classes, input_dim, dropout_prob=0.0, learning_rate=1e-3,  \
                 param_initializer_range=0.02, phase='train'):
        
        """  
        Args:
            phase: train, eval, pred
            lang: en, ch, ...
        """

        self._is_training = phase == 'train'
        self._hidden_size = input_dim

        self.num_classes = num_classes
    
        self._dropout_prob = dropout_prob if phase == 'train' else 0.0
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=param_initializer_range)

        self.learning_rate = learning_rate
        self._preds = []


    @property
    def inputs_attrs(self):
        reader = {}
        bb = {"encoder_outputs": [[-1, -1, -1], 'float32']}
        if self._is_training:
            reader["label_ids"] = [[-1, -1], 'int64']
            reader["seq_lens"] = [[-1], 'int64']
        return {'reader': reader, 'backbone': bb}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {'loss': [[1], 'float32']}
        else:
            return {'emission': [[-1, self.num_classes], 'float32']}

    def build(self, inputs, scope_name=''):
        token_emb = inputs['backbone']['encoder_outputs']
        if self._is_training:
            label_ids = inputs['reader']['label_ids']
            seq_lens = inputs['reader']['seq_lens']

        emission = fluid.layers.fc(
            size=self.num_classes,
            input=token_emb,
            param_attr=fluid.ParamAttr(
                initializer=self._param_initializer,
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)),
            bias_attr=fluid.ParamAttr(
                name=scope_name+"cls_out_b", initializer=fluid.initializer.Constant(0.)),
            num_flatten_dims=2)

        if self._is_training:

            # compute loss
            crf_cost = fluid.layers.linear_chain_crf(  
                input=emission,
                label=label_ids,
                param_attr=fluid.ParamAttr(
                    name=scope_name+'crfw', learning_rate=self.learning_rate),
                length=seq_lens)

            avg_cost = fluid.layers.mean(x=crf_cost)
            crf_decode = fluid.layers.crf_decoding(
                input=emission,
                param_attr=fluid.ParamAttr(name=scope_name+'crfw'),
                length=seq_lens)

            (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
            num_correct_chunks) = fluid.layers.chunk_eval(
                input=crf_decode,
                label=label_ids,
                chunk_scheme="IOB",
                num_chunk_types=int(math.ceil((self.num_classes - 1) / 2.0)),
                seq_length=seq_lens)
            chunk_evaluator = fluid.metrics.ChunkEvaluator()
            chunk_evaluator.reset()

            return {"loss": avg_cost}
        else:
            return {"emission": emission} 

    def batch_postprocess(self, rt_outputs):
        if not self._is_training:
            emission = rt_outputs['emission']
            preds = np.argmax(emission, -1)
            self._preds.extend(preds.tolist())

    def epoch_postprocess(self, post_inputs, output_dir=None):
        # there is no post_inputs needed and not declared in epoch_inputs_attrs, hence no elements exist in post_inputs
        if not self._is_training:
            if output_dir is None:
                raise ValueError('argument output_dir not found in config. Please add it into config dict/file.')
            with open(os.path.join(output_dir, 'predictions.json'), 'w') as writer:
                for p in self._preds:
                    writer.write(str(p)+'\n')
            print('Predictions saved at '+os.path.join(output_dir, 'predictions.json'))
