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


class task(object):

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

