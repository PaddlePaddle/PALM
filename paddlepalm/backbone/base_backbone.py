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


class Backbone(object):
    """interface of backbone model."""

    def __init__(self, phase):
        """该函数完成一个主干网络的构造，至少需要包含一个phase参数。
        注意：实现该构造函数时，必须保证对基类构造函数的调用，以创建必要的框架内建的成员变量。
        Args:
            phase: str类型。用于区分主干网络被调用时所处的运行阶段，目前支持训练阶段train和预测阶段predict
            """

        assert isinstance(config, dict)

    @property
    def inputs_attr(self):
        """描述backbone从reader处需要得到的输入对象的属性，包含各个对象的名字、shape以及数据类型。当某个对象
        为标量数据类型（如str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape
        中的相应维度设置为-1。

        Return:
            dict类型。对各个输入对象的属性描述。例如，
            对于文本分类和匹配任务，bert backbone依赖的reader对象主要包含如下的对象
                {"token_ids": ([-1, max_len], 'int64'),
                 "input_ids": ([-1, max_len], 'int64'),
                 "segment_ids": ([-1, max_len], 'int64'),
                 "input_mask": ([-1, max_len], 'float32')}"""
        raise NotImplementedError()

    @property
    def outputs_attr(self):
        """描述backbone输出对象的属性，包含各个对象的名字、shape以及数据类型。当某个对象为标量数据类型（如
        str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape中的相应维度设置为-1。
        
        Return:
            dict类型。对各个输出对象的属性描述。例如，
            对于文本分类和匹配任务，bert backbone的输出内容可能包含如下的对象
                {"word_emb": ([-1, max_seqlen, word_emb_size], 'float32'),
                 "sentence_emb": ([-1, hidden_size], 'float32'),
                 "sim_vec": ([-1, hidden_size], 'float32')}""" 
        raise NotImplementedError()

    def build(self, inputs):
        """建立backbone的计算图。将符合inputs_attr描述的静态图Variable输入映射成符合outputs_attr描述的静态图Variable输出。
        Args:
            inputs: dict类型。字典中包含inputs_attr中的对象名到计算图Variable的映射，inputs中至少会包含inputs_attr中定义的对象
        Return:
           需要输出的计算图变量，输出对象会被加入到fetch_list中，从而在每个训练/推理step时得到runtime的计算结果，该计算结果会被传入postprocess方法中供用户处理。
            """
        raise NotImplementedError()
