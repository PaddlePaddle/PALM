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
from paddlepalm.head.base_head import Head
from paddle.fluid import layers
import numpy as np
import os
from paddlepalm.backbone.utils.transformer import pre_process_layer

class MaskLM(Head):
    '''
    mlm
    '''
    def __init__(self, input_dim, vocab_size, hidden_act, dropout_prob=0.0, \
                 param_initializer_range=0.02, phase='train'):
        self._is_training = phase == 'train'
        self._emb_size = input_dim
        self._hidden_size = input_dim
        self._dropout_prob = dropout_prob if phase == 'train' else 0.0
        self._preds = []

        self._vocab_size = vocab_size
        self._hidden_act = hidden_act
        self._initializer_range = param_initializer_range
    
    @property
    def inputs_attrs(self):
        reader = {
            "mask_label": [[-1], 'int64'],
            "mask_pos": [[-1], 'int64'],
            }
        if not self._is_training:
            del reader['mask_label']
        bb = {
            "encoder_outputs": [[-1, -1, self._hidden_size], 'float32'],
            "embedding_table": [[-1, self._vocab_size, self._emb_size], 'float32']}
        return {'reader': reader, 'backbone': bb}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {"loss": [[1], 'float32']}
        else:
            return {"logits": [[-1], 'float32']}

    def build(self, inputs, scope_name=""):
        mask_pos = inputs["reader"]["mask_pos"]
        
        word_emb = inputs["backbone"]["embedding_table"]
        enc_out = inputs["backbone"]["encoder_outputs"]

        if self._is_training:
            mask_label = inputs["reader"]["mask_label"]
            l1 = enc_out.shape[0] 
            l2 = enc_out.shape[1]
            bxs = fluid.layers.fill_constant(shape=[1], value=l1*l2, dtype='int64')
            max_position = bxs - 1
            mask_pos = fluid.layers.elementwise_min(mask_pos, max_position)
            mask_pos.stop_gradient = True

        emb_size = word_emb.shape[-1]

        _param_initializer = fluid.initializer.TruncatedNormal(
            scale=self._initializer_range)

        reshaped_emb_out = fluid.layers.reshape(
            x=enc_out, shape=[-1, emb_size])

        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name=scope_name+'mask_lm_trans_fc.w_0',
                initializer=_param_initializer),
                bias_attr=fluid.ParamAttr(name=scope_name+'mask_lm_trans_fc.b_0'))
        # transform: layer norm
        mask_trans_feat = pre_process_layer(
            mask_trans_feat, 'n', name=scope_name+'mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name=scope_name+"mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        fc_out = fluid.layers.matmul(
            x=mask_trans_feat,
            y=word_emb,
            transpose_y=True)
        fc_out += fluid.layers.create_parameter(
            shape=[self._vocab_size],
            dtype='float32',
            attr=mask_lm_out_bias_attr,
            is_bias=True)

        if self._is_training:
            inputs = fluid.layers.softmax(fc_out)
            mask_lm_loss = fluid.layers.cross_entropy(
                input=inputs, label=mask_label)
            loss = fluid.layers.mean(mask_lm_loss)
            return {'loss': loss}
        else:
            return {'logits': fc_out}

    def batch_postprocess(self, rt_outputs):
        if not self._is_training:
            logits = rt_outputs['logits']
            preds = np.argmax(logits, -1)
            self._preds.extend(preds.tolist())
            return preds

    def epoch_postprocess(self, post_inputs, output_dir=None):
        # there is no post_inputs needed and not declared in epoch_inputs_attrs, hence no elements exist in post_inputs
        if not self._is_training:
            if output_dir is None:
                for p in self._preds:
                    print(p)
            else:
                with open(os.path.join(output_dir, 'predictions.json'), 'w') as writer:
                    for p in self._preds:
                        writer.write(str(p)+'\n')
                print('Predictions saved at '+os.path.join(output_dir, 'predictions.json'))


