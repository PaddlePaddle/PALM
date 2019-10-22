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
    matching
    '''
    def __init__(self, config, phase, backbone_config=None):
        self._is_training = phase == 'train'
        self._hidden_size = backbone_config['hidden_size']
        self._vocab_size = backbone_config['vocab_size']
        self._hidden_act = backbone_config['hidden_act']
        self._initializer_range = backbone_config['initializer_range']
    
    @property
    def inputs_attrs(self):
        if self._is_training:
            reader = {"label_ids": [[-1, 1], 'int64']}
        else:
            reader = {}
        bb = {"encoder_outputs": [[-1, self._hidden_size], 'float32']}
        return {'reader': reader, 'backbone': bb}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {"loss": [[1], 'float32']}
        else:
            return {"logits": [[-1, 1], 'float32']}

    def build(self, inputs):
        mask_label = inputs["reader"]["mask_label"] 
        mask_pos = inputs["reader"]["mask_pos"] 
        word_emb = inputs["backbone"]["word_embedding"]
        enc_out = inputs["backbone"]["encoder_outputs"]

        emb_size = word_emb.shape[-1]

        _param_initializer = fluid.initializer.TruncatedNormal(
            scale=self._initializer_range)

        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        reshaped_emb_out = fluid.layers.reshape(
            x=enc_out, shape=[-1, emb_size])

        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)
        num_seqs = fluid.layers.fill_constant(shape=[1], value=512, dtype='int64')

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=_param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        # transform: layer norm
        mask_trans_feat = pre_process_layer(
            mask_trans_feat, 'n', name='mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        # print fluid.default_main_program().global_block()

        # fc_out = fluid.layers.matmul(
        #     x=mask_trans_feat,
        #     y=fluid.default_main_program().global_block().var(
        #         _word_emb_name),
        #     transpose_y=True)

        fc_out = fluid.layers.matmul(
            x=mask_trans_feat,
            y=word_emb,
            transpose_y=True)
        fc_out += fluid.layers.create_parameter(
            shape=[self._vocab_size],
            dtype='float32',
            attr=mask_lm_out_bias_attr,
            is_bias=True)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        loss = fluid.layers.mean(mask_lm_loss)

        if self._is_training:
            return {'loss': loss}
        else:
            return None


