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
"""BERT model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid import layers

import backbone.utils.transformer as transformer
    
class Model(object):
    
    def __init__(self,
                 config,
                 is_training=False,
                 model_name=''):

        self._emb_size = config["hidden_size"]
        self._n_layer = config["num_hidden_layers"]
        self._n_head = config["num_attention_heads"]
        self._voc_size = config["vocab_size"]
        self._max_position_seq_len = config["max_position_embeddings"]
        self._sent_types = config["type_vocab_size"]
        self._hidden_act = config["hidden_act"]
        self._prepostprocess_dropout = config["hidden_dropout_prob"]
        self._attention_dropout = config["attention_probs_dropout_prob"]

        self._is_training = is_training

        self.model_name = model_name

        self._word_emb_name = self.model_name + "word_embedding"
        self._pos_emb_name = self.model_name + "pos_embedding"
        self._sent_emb_name = self.model_name + "sent_embedding"

        # Initialize all weigths by truncated normal initializer, and all biases 
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config["initializer_range"])

    def build_model(self, reader_input, use_fp16=False):
        
        dtype = "float16" if use_fp16 else "float32"

        src_ids, pos_ids, sent_ids, input_mask = reader_input[:4]
        # padding id in vocabulary must be set to 0
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        
        self.emb_out = emb_out
        
        position_emb_out = fluid.layers.embedding(
            input=pos_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=dtype,
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer))
    
        self.position_emb_out = position_emb_out

        sent_emb_out = fluid.layers.embedding(
            sent_ids,
            size=[self._sent_types, self._emb_size],
            dtype=dtype,
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer))

        self.sent_emb_out = sent_emb_out

        emb_out = emb_out + position_emb_out + sent_emb_out

        emb_out = transformer.pre_process_layer(
            emb_out, 'nd', self._prepostprocess_dropout, name='pre_encoder')

        if dtype == "float16":
            input_mask = fluid.layers.cast(x=input_mask, dtype=dtype)

        self_attn_mask = fluid.layers.matmul(
            x = input_mask, y = input_mask, transpose_y = True)

        self_attn_mask = fluid.layers.scale(
            x = self_attn_mask, scale = 10000.0, bias = -1.0, bias_after_scale = False)
        
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out = transformer.encoder(
            enc_input = emb_out,
            attn_bias = n_head_self_attn_mask,
            n_layer = self._n_layer,
            n_head = self._n_head,
            d_key = self._emb_size // self._n_head,
            d_value = self._emb_size // self._n_head,
            d_model = self._emb_size,
            d_inner_hid = self._emb_size * 4,
            prepostprocess_dropout = self._prepostprocess_dropout,
            attention_dropout = self._attention_dropout,
            relu_dropout = 0,
            hidden_act = self._hidden_act,
            preprocess_cmd = "",
            postprocess_cmd = "dan",
            param_initializer = self._param_initializer,
            name = self.model_name + 'encoder')

        next_sent_feat = fluid.layers.slice(
            input = self._enc_out, axes = [1], starts = [0], ends = [1])
        self.next_sent_feat = fluid.layers.fc(
            input = next_sent_feat,
            size = self._emb_size,
            act = "tanh",
            param_attr = fluid.ParamAttr(
                name = self.model_name + "pooled_fc.w_0", 
                initializer = self._param_initializer),
            bias_attr = "pooled_fc.b_0")
    @property
    def final_word_representation(self):
        """final layer output of transformer encoder as the (contextual) word representation"""
        return self._enc_out

    @property
    def final_sentence_representation(self):
        """final representation of the first token ([CLS]) as sentence representation """

        return self.next_sent_feat


if __name__ == "__main__":
    print("hello world!")


