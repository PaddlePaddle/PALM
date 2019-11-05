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
"""v1.1"""

class reader(object):
    """interface of data manager."""

    def __init__(self, config):
        assert isinstance(config, dict)

    # @property
    # def inputs_attr(self):
    #     """描述reader输入对象的属性，包含各个对象的名字、shape以及数据类型。当某个对象为标量数据类型（如str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape中的相应维度设置为-1.
    #     Return:
    #         dict类型。对各个输入对象的属性描述。例如，
    #         对于文本分类任务，可能需要包含输入文本和所属标签的id
    #             {"text": ([], 'str'),
    #              "label": ([], 'int')}
    #         对于标注任务，可能需要输入词序列和对应的标签
    #             {"tokens", ([-1], 'str'),
    #              "tags", ([-1], 'str')}
    #         对于机器阅读理解任务，可能需要包含上下文、问题、回答、答案区域的起止位置等
    #             {"paragraph", ([], 'str'),
    #              "question", ([], 'str'),
    #              "start_position", ([], 'int')
    #         """
    #     raise NotImplementedError()

    @property
    def outputs_attr(self):
        """描述reader输出对象（被yield出的对象）的属性，包含各个对象的名字、shape以及数据类型。当某个对象为标量数据类型（如str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape中的相应维度设置为-1。
        注意：当使用mini-batch梯度下降学习策略时，，应为常规的输入对象设置batch_size维度（一般为-1）
        Return:
            dict类型。对各个输入对象的属性描述。例如，
            对于文本分类和匹配任务，yield的输出内容可能包含如下的对象（下游backbone和task可按需访问其中的对象）
                {"token_ids": ([-1, max_len], 'int64'),
                 "input_ids": ([-1, max_len], 'int64'),
                 "segment_ids": ([-1, max_len], 'int64'),
                 "input_mask": ([-1, max_len], 'float32'),
                 "label": ([-1], 'int')}
        """
        raise NotImplementedError()

    # def parse_line(self):
    #     """框架内部使用字典描述每个样本，字典的key为inputs_attr，value为每个input对应的符合attr描述的值。
    #         该函数负责将文本行解析成符合inputs_attr描述的字典类型的样本。默认的parse_line方法会读取json格式的数据集文件，数据集的每一行为json格式描述的样本。
    #         用户可通过对该方法的继承改写来适配不同格式的数据集，例如csv格式甚至tfrecord文件。
    #         """
    #     raise NotImplementedError()
    # 
    # def tokenize(self, line):
    #     """框架中内置了word piece tokenizer等分词器，用户可通过修改tokenizer超参数来制定使用的分词器，若内置的分词器均无法满足需求，用户可通过对该方法的继承改写来自定义分词器。
    #         Args:
    #             - line: a unicode string. 
    #         Return:
    #             a list of tokens
    #         """
    #     raise NotImplementedError()
    
    def iterator(self):
        """数据集遍历接口，注意，当数据集遍历到尾部时该接口应自动完成指针重置，即重新从数据集头部开始新的遍历。
        Yield:
            (dict) elements that meet the requirements in output_templete
        """
        raise NotImplementedError()

    @property
    def num_examples(self):
        """数据集中的样本数量，即每个epoch中iterator所生成的样本数。注意，使用滑动窗口等可能导致数据集样本数发生变化的策略时，该接口应返回runtime阶段的实际样本数。"""
        raise NotImplementedError()



class backbone(object):
    """interface of backbone model."""

    def __init__(self, config, phase):
        """
        Args:
            config: dict类型。描述了 多任务配置文件+预训练模型配置文件 中定义超参数
            phase: str类型。运行阶段，目前支持train和predict
            """
        assert isinstance(config, dict)

    @property
    def inputs_attr(self):
        """描述backbone从reader处需要得到的输入对象的属性，包含各个对象的名字、shape以及数据类型。当某个对象为标量数据类型（如str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape中的相应维度设置为-1。
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
        """描述backbone输出对象的属性，包含各个对象的名字、shape以及数据类型。当某个对象为标量数据类型（如str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape中的相应维度设置为-1。
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




class task_paradigm(object):

    def __init__(self, config, phase, backbone_config):
        """
            config: dict类型。描述了 任务实例(task instance)+多任务配置文件 中定义超参数
            phase: str类型。运行阶段，目前支持train和predict
            """

    @property
    def inputs_attrs(self):
        """描述task_layer需要从reader, backbone等输入对象集合所读取到的输入对象的属性，第一级key为对象集和的名字，如backbone，reader等（后续会支持更灵活的输入），第二级key为对象集和中各对象的属性，包括对象的名字，shape和dtype。当某个对象为标量数据类型（如str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape中的相应维度设置为-1。
        Return:
            dict类型。对各个对象集及其输入对象的属性描述。"""
        raise NotImplementedError()

    @property
    def outputs_attr(self):
        """描述task输出对象的属性，包括对象的名字，shape和dtype。输出对象会被加入到fetch_list中，从而在每个训练/推理step时得到runtime的计算结果，该计算结果会被传入postprocess方法中供用户处理。
        当某个对象为标量数据类型（如str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape中的相应维度设置为-1。
        Return:
            dict类型。对各个输入对象的属性描述。注意，训练阶段必须包含名为loss的输出对象。
            """

        raise NotImplementedError()

    @property
    def epoch_inputs_attrs(self):
        return {}

    def build(self, inputs, scope_name=""):
        """建立task_layer的计算图。将符合inputs_attrs描述的来自各个对象集的静态图Variables映射成符合outputs_attr描述的静态图Variable输出。
        Args:
            inputs: dict类型。字典中包含inputs_attrs中的对象名到计算图Variable的映射，inputs中至少会包含inputs_attr中定义的对象
        Return:
           需要输出的计算图变量，输出对象会被加入到fetch_list中，从而在每个训练/推理step时得到runtime的计算结果，该计算结果会被传入postprocess方法中供用户处理。

        """
        raise NotImplementedError()

    def postprocess(self, rt_outputs):
        """每个训练或推理step后针对当前batch的task_layer的runtime计算结果进行相关后处理。注意，rt_outputs除了包含build方法，还自动包含了loss的计算结果。"""
        pass
        
    def epoch_postprocess(self, post_inputs):
        pass

