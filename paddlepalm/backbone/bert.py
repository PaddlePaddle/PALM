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
"""v1.1 
BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from paddle.fluid import layers

from paddlepalm.backbone.utils.transformer import pre_process_layer, encoder
from paddlepalm.backbone.base_backbone import Backbone


class BERT(Backbone):


    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, vocab_size, \
          max_position_embeddings, type_vocab_size, hidden_act, hidden_dropout_prob, \
          attention_probs_dropout_prob, initializer_range, is_pairwise=False, phase='train'):
     
        self._emb_size = hidden_size
        self._n_layer = num_hidden_layers
        self._n_head = num_attention_heads
        self._voc_size = vocab_size
        self._max_position_seq_len = max_position_embeddings
        self._sent_types = type_vocab_size

       
        self._hidden_act = hidden_act
        self._prepostprocess_dropout = hidden_dropout_prob
        self._attention_dropout = attention_probs_dropout_prob

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._task_emb_name = "task_embedding"
        self._emb_dtype = "float32"
        self._phase = phase
        self._is_pairwise = is_pairwise
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=initializer_range)

    @classmethod
    def from_config(self, config, phase='train'):
        
        assert 'hidden_size' in config, "{} is required to initialize ERNIE".format('')
        assert 'num_hidden_layers' in config, "{} is required to initialize ERNIE".format('num_hidden_layers')
        assert 'num_attention_heads' in config, "{} is required to initialize ERNIE".format('num_attention_heads')
        assert 'vocab_size' in config, "{} is required to initialize ERNIE".format('vocab_size')
        assert 'max_position_embeddings' in config, "{} is required to initialize ERNIE".format('max_position_embeddings')
        assert 'sent_type_vocab_size' in config or 'type_vocab_size' in config, \
            "{} is required to initialize ERNIE".format('type_vocab_size')
        assert 'hidden_act' in config, "{} is required to initialize ERNIE".format('hidden_act')
        assert 'hidden_dropout_prob' in config, "{} is required to initialize ERNIE".format('hidden_dropout_prob')
        assert 'attention_probs_dropout_prob' in config, \
            "{} is required to initialize ERNIE".format('attention_probs_dropout_prob')
        assert 'initializer_range' in config, "{} is required to initialize ERNIE".format('initializer_range')

        hidden_size = config['hidden_size']
        num_hidden_layers = config['num_hidden_layers']
        num_attention_heads = config['num_attention_heads']
        vocab_size = config['vocab_size']
        max_position_embeddings = config['max_position_embeddings']
        if 'sent_type_vocab_size' in config:
            sent_type_vocab_size = config['sent_type_vocab_size']
        else:
            sent_type_vocab_size = config['type_vocab_size']

        hidden_act = config['hidden_act']
        hidden_dropout_prob = config['hidden_dropout_prob']
        attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        initializer_range = config['initializer_range']
        if 'is_pairwise' in config:
            is_pairwise = config['is_pairwise']
        else:
            is_pairwise = False

        return self(hidden_size, num_hidden_layers, num_attention_heads, vocab_size, \
          max_position_embeddings, sent_type_vocab_size, \
          hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, initializer_range, is_pairwise, phase)

    @property
    def inputs_attr(self):
        ret = {"token_ids": [[-1, -1], 'int64'],
               "position_ids": [[-1, -1], 'int64'],
               "segment_ids": [[-1, -1], 'int64'],
               "input_mask": [[-1, -1, 1], 'float32'],
               }
        if self._is_pairwise and self._phase=='train':
            ret.update({"token_ids_neg": [[-1, -1], 'int64'],
                        "position_ids_neg": [[-1, -1], 'int64'],
                        "segment_ids_neg": [[-1, -1], 'int64'],
                        "input_mask_neg": [[-1, -1, 1], 'float32'],
                        })
        return ret

    @property
    def outputs_attr(self):
        ret = {"word_embedding": [[-1, -1, self._emb_size], 'float32'],
               "embedding_table": [[-1, self._voc_size, self._emb_size], 'float32'],
               "encoder_outputs": [[-1, -1, self._emb_size], 'float32'],
               "sentence_embedding": [[-1, self._emb_size], 'float32'],
               "sentence_pair_embedding": [[-1, self._emb_size], 'float32']}
        if self._is_pairwise and self._phase == 'train':
            ret.update({"word_embedding_neg": [[-1, -1, self._emb_size], 'float32'],
                        "encoder_outputs_neg": [[-1, -1, self._emb_size], 'float32'],
                        "sentence_embedding_neg": [[-1, self._emb_size], 'float32'],
                        "sentence_pair_embedding_neg": [[-1, self._emb_size], 'float32']})
        return ret 

    def build(self, inputs, scope_name=""):
        src_ids = inputs['token_ids']
        pos_ids = inputs['position_ids']
        sent_ids = inputs['segment_ids']
        input_mask = inputs['input_mask']

        self._emb_dtype = 'float32'

        input_buffer = {}
        output_buffer = {}
        input_buffer['base'] = [src_ids, pos_ids, sent_ids, input_mask]
        output_buffer['base'] = {}

        if self._is_pairwise and self._phase =='train':
            src_ids = inputs['token_ids_neg']
            pos_ids = inputs['position_ids_neg']
            sent_ids = inputs['segment_ids_neg']
            input_mask = inputs['input_mask_neg']
            input_buffer['neg'] = [src_ids, pos_ids, sent_ids, input_mask]
            output_buffer['neg'] = {}
        
        for key, (src_ids, pos_ids, sent_ids, input_mask) in input_buffer.items():
            # padding id in vocabulary must be set to 0
            emb_out = fluid.embedding(
                input=src_ids,
                size=[self._voc_size, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=scope_name+self._word_emb_name, initializer=self._param_initializer),
                is_sparse=False)

            # fluid.global_scope().find_var('backbone-word_embedding').get_tensor()
            embedding_table = fluid.default_main_program().global_block().var(scope_name+self._word_emb_name)
            
            position_emb_out = fluid.embedding(
                input=pos_ids,
                size=[self._max_position_seq_len, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=scope_name+self._pos_emb_name, initializer=self._param_initializer))

            sent_emb_out = fluid.embedding(
                sent_ids,
                size=[self._sent_types, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(
                    name=scope_name+self._sent_emb_name, initializer=self._param_initializer))

            emb_out = emb_out + position_emb_out
            emb_out = emb_out + sent_emb_out

            emb_out = pre_process_layer(
                emb_out, 'nd', self._prepostprocess_dropout, name=scope_name+'pre_encoder')

            self_attn_mask = fluid.layers.matmul(
                x=input_mask, y=input_mask, transpose_y=True)

            self_attn_mask = fluid.layers.scale(
                x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
            n_head_self_attn_mask = fluid.layers.stack(
                x=[self_attn_mask] * self._n_head, axis=1)
            n_head_self_attn_mask.stop_gradient = True

            enc_out = encoder(
                enc_input=emb_out,
                attn_bias=n_head_self_attn_mask,
                n_layer=self._n_layer,
                n_head=self._n_head,
                d_key=self._emb_size // self._n_head,
                d_value=self._emb_size // self._n_head,
                d_model=self._emb_size,
                d_inner_hid=self._emb_size * 4,
                prepostprocess_dropout=self._prepostprocess_dropout,
                attention_dropout=self._attention_dropout,
                relu_dropout=0,
                hidden_act=self._hidden_act,
                preprocess_cmd="",
                postprocess_cmd="dan",
                param_initializer=self._param_initializer,
                name=scope_name+'encoder')

            
            next_sent_feat = fluid.layers.slice(
                input=enc_out, axes=[1], starts=[0], ends=[1])
            next_sent_feat = fluid.layers.reshape(next_sent_feat, [-1, next_sent_feat.shape[-1]])
            next_sent_feat = fluid.layers.fc(
                input=next_sent_feat,
                size=self._emb_size,
                act="tanh",
                param_attr=fluid.ParamAttr(
                    name=scope_name+"pooled_fc.w_0", initializer=self._param_initializer),
                bias_attr=scope_name+"pooled_fc.b_0")
            output_buffer[key]['word_embedding'] = emb_out
            output_buffer[key]['encoder_outputs'] = enc_out
            output_buffer[key]['sentence_embedding'] = next_sent_feat
            output_buffer[key]['sentence_pair_embedding'] = next_sent_feat
        
        ret = {}
        ret['embedding_table'] = embedding_table
        ret['word_embedding'] = output_buffer['base']['word_embedding']
        ret['encoder_outputs'] = output_buffer['base']['encoder_outputs']
        ret['sentence_embedding'] = output_buffer['base']['sentence_embedding']
        ret['sentence_pair_embedding'] = output_buffer['base']['sentence_pair_embedding']

        if self._is_pairwise and self._phase == 'train':
            ret['word_embedding_neg'] = output_buffer['neg']['word_embedding']
            ret['encoder_outputs_neg'] = output_buffer['neg']['encoder_outputs']
            ret['sentence_embedding_neg'] = output_buffer['neg']['sentence_embedding']
            ret['sentence_pair_embedding_neg'] = output_buffer['neg']['sentence_pair_embedding']
        
        return ret
                    
    def postprocess(self, rt_outputs):
        pass


class Model(BERT):
    """BERT wrapper for ConfigController"""
    def __init__(self, config, phase):
        BERT.from_config(config, phase=phase)


