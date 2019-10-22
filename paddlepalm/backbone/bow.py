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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from paddle.fluid import layers

class Model(backbone):
    
    def __init__(self, config, phase):

        # self._is_training = phase == 'train' # backbone一般不用关心运行阶段，因为outputs在任何阶段基本不会变

        self._emb_size = config["emb_size"]
        self._voc_size = config["vocab_size"]

    @property
    def inputs_attr(self):
        return {"token_ids": [-1, self._max_position_seq_len, 1], 'int64']}

    @property
    def outputs_attr(self):
        return {"word_emb": [-1, self._max_position_seq_len, self._emb_size],
                "sentence_emb": [-1, self._emb_size*2]}

    def build(self, inputs):

        tok_ids = inputs['token_ids']
        
        emb_out = layers.embedding(
            input=tok_ids,
            size=[self._voc_size, self._emb_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='word_emb', 
                initializer=fluid.initializer.TruncatedNormal(scale=0.1)),
            is_sparse=False)

        sent_emb1 = layers.reduce_mean(emb_out, axis=1)
        sent_emb2 = layers.reduce_max(emb_out, axis=1)
        sent_emb = layers.concat([sent_emb1, sent_emb2], axis=1)
        return {'word_emb': emb_out,
                'sentence_emb': sent_emb}

    def postprocess(self, rt_outputs):
        pass


