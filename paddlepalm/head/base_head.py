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

import os
import json
import copy

class Head(object):

    def __init__(self, phase='train'):
        """该函数完成一个任务头的构造，至少需要包含一个phase参数。
        注意：实现该构造函数时，必须保证对基类构造函数的调用，以创建必要的框架内建的成员变量。
        Args:
            phase: str类型。用于区分任务头被调用时所处的任务运行阶段，目前支持训练阶段train和预测阶段predict
            """
        self._stop_gradient = {}
        self._phase = phase
        self._prog = None
        self._results_buffer = []

    @property
    def inputs_attrs(self):
        """step级别的任务输入对象声明。

        描述该任务头所依赖的reader、backbone和来自其他任务头的输出对象（每个step获取一次）。使用字典进行描述，
        字典的key为输出对象所在的组件（如’reader‘，’backbone‘等），value为该组件下任务头所需要的输出对象集。
        输出对象集使用字典描述，key为输出对象的名字（该名字需保证在相关组件的输出对象集中），value为该输出对象
        的shape和dtype。当某个输出对象的某个维度长度可变时，shape中的相应维度设置为-1。

        Return:
            dict类型。描述该任务头所依赖的step级输入，即来自各个组件的输出对象。"""
        raise NotImplementedError()

    @property
    def outputs_attr(self):
        """step级别的任务输出对象声明。

        描述该任务头的输出对象（每个step输出一次），包括每个输出对象的名字，shape和dtype。输出对象会被加入到
        fetch_list中，从而在每个训练/推理step时得到实时的计算结果，该计算结果可以传入batch_postprocess方
        法中进行当前step的后处理。当某个对象为标量数据类型（如str, int, float等）时，shape设置为空列表[]，
        当某个对象的某个维度长度可变时，shape中的相应维度设置为-1。

        Return:
            dict类型。描述该任务头所产生的输出对象。注意，在训练阶段时必须包含名为loss的输出对象。
            """

        raise NotImplementedError()

    @property
    def epoch_inputs_attrs(self):
        """epoch级别的任务输入对象声明。

        描述该任务所依赖的来自reader、backbone和来自其他任务头的输出对象（每个epoch结束后产生一次），如完整的
        样本集，有效的样本数等。使用字典进行描述，字典的key为输出对象所在的组件（如’reader‘，’backbone‘等），
        value为该组件下任务头所需要的输出对象集。输出对象集使用字典描述，key为输出对象的名字（该名字需保证在相关
        组件的输出对象集中），value为该输出对象的shape和dtype。当某个输出对象的某个维度长度可变时，shape中的相
        应维度设置为-1。
        
        Return:
            dict类型。描述该任务头所产生的输出对象。注意，在训练阶段时必须包含名为loss的输出对象。
        """
        return {}

    def build(self, inputs, scope_name=""):
        """建立任务头的计算图。

        将符合inputs_attrs描述的来自各个对象集的静态图Variables映射成符合outputs_attr描述的静态图Variable输出。

        Args:
            inputs: dict类型。字典中包含inputs_attrs中的对象名到计算图Variable的映射，inputs中至少会包含inputs_attr中定义的对象
        Return:
           需要输出的计算图变量，输出对象会被加入到fetch_list中，从而在每个训练/推理step时得到runtime的计算结果，该计算结果会被传入postprocess方法中供用户处理。
        """
        raise NotImplementedError()

    def batch_postprocess(self, rt_outputs):
        """batch/step级别的后处理。

        每个训练或推理step后针对当前batch的任务头输出对象的实时计算结果来进行相关后处理。
        默认将输出结果存储到缓冲区self._results_buffer中。"""
        if isinstance(rt_outputs, dict):
            keys = rt_outputs.keys()
            vals = [rt_outputs[k] for k in keys]
            lens = [len(v) for v in vals]
            if len(set(lens)) == 1:
                results = [dict(zip(*[keys, i])) for i in zip(*vals)]
                self._results_buffer.extend(results)
                return results
            else:
                print('WARNING: irregular output results. visualize failed.')
                self._results_buffer.append(rt_outputs)
        return None

    def reset(self):
        """清空该任务头的缓冲区（在训练或推理过程中积累的处理结果）"""
        self._results_buffer = []

    def get_results(self):
        """返回当前任务头积累的处理结果。"""
        return copy.deepcopy(self._results_buffer)
        
    def epoch_postprocess(self, post_inputs=None, output_dir=None):
        """epoch级别的后处理。

        每个训练或推理epoch结束后，对积累的各样本的后处理结果results进行后处理。默认情况下，当output_dir为None时，直接将results打印到
        屏幕上。当指定output_dir时，将results存储在指定的文件夹内，并以任务头所处阶段来作为存储文件的文件名。

        Args:
            post_inputs: 当声明的epoch_inputs_attr不为空时，该参数会携带对应的输入变量的内容。
            output_dir: 积累结果的保存路径。
        """
        if output_dir is not None:
            for i in self._results_buffer:
                print(i)
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, self._phase), 'w') as writer:
                for i in self._results_buffer:
                    writer.write(json.dumps(i)+'\n')
