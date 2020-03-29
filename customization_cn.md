# PALM组件定制化教程

PALM支持对如下组件自定义：

- **head**
  定义一个新的任务输出头，接收来自backbone和reader的输入，输出训练阶段的loss和预测阶段的预测结果。例如：分类任务头，序列标注任务头，机器阅读理解任务头等。
- **backbone**
  定义一个新的主干网络，接收来自reader的文本相关的序列特征输入（如token ids），输出文本的特征向量表示（如词向量、上下文相关的词向量表示、句子向量等）。例如：BERT encoder，CNN encoder等。
- **reader**
  定义一个新的数据集载入与预处理模块，接收来自原始数据集文件的输入（纯文本，原始标签等），输出文本相关的序列特征（如token ids，position ids等）。例如：文本分类数据集处理模块；文本匹配数据集处理模块等。
- **optimizer**
  定义一个新的优化器
- **lr_sched**
  定义一种新的学习率规划策略

PALM中的每个组件均使用类来描述，因此可以允许存在内部记忆（成员变量）。

新增某种类型的组件时，只需要实现该组件类型所在目录下的接口类中所描述的方法。若希望新增的组件跟框架的某个内置组件功能相似，那么实现新增组件时，可以继承自已有的内置组件，且仅对需要变动的方法进行修改即可。

### head自定义

head的接口类（Interface）位于`paddlepalm/head/base_head.py`。

该接口类定义如下：

```python
# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
```



在基类的基础上，定义一个全新的Head时需要至少实现的方法有：

- \_\_init\_\_
- inputs_attrs
- outputs_attr
- build

可以重写的方法有：

- epoch_inputs_attrs
- batch_postprocess
- epoch_postprocess

### backbone自定义

backbone的接口类（Interface）位于`paddlepalm/backbone/base_backbone.py`。

该接口类定义如下：

```python
# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
```



在基类的基础上，定义一个全新的Backbone时需要至少实现的方法有：

- \_\_init\_\_
- input_attrs
- output_attr
- build

### reader自定义

reader的接口类（Interface）位于`paddlepalm/reader/base_reader.py`。

该接口类定义如下：

```python
# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import copy

class Reader(object):
    """interface of data reader."""

    def __init__(self, phase='train'):
        """该函数完成一个Reader的构造，至少需要包含一个phase参数。
        注意：实现该构造函数时，必须保证对基类构造函数的调用，以创建必要的框架内建的成员变量。
        Args:
            phase: str类型。用于区分主干网络被调用时所处的运行阶段，目前支持训练阶段train和预测阶段predict
            """
        
        self._phase = phase
        self._batch_size = None
        self._num_epochs = 1
        self._register = set()
        self._registered_backbone = None

    @classmethod
    def create_register(self):
        return set()
        
    def clone(self, phase='train'):
        """拷贝一个新的reader对象。"""
        if phase == self._phase:
            return copy(self)
        else:
            ret = copy(self)
            ret._phase = phase
            return ret

    def require_attr(self, attr_name):
        """在注册器中新增一个需要产生的对象。

        Args:
            attr_name: 需要产出的对象的对象名，例如’segment_ids‘。
            """
        self._register.add(attr_name)
            
    def register_with(self, backbone):
        """根据backbone对输入对象的依赖，在注册器中对每个依赖的输入对象进行注册。

        Args:
            backbone: 需要对接的主干网络。
        """
        for attr in backbone.inputs_attr:
            self.require_attr(attr)
        self._registered_backbone = backbone

    def get_registered_backbone(self):
        """返回该reader所注册的backbone。"""
        return self._registered_backbone

    def _get_registed_attrs(self, attrs):
        ret = {}
        for i in self._register:
            if i not in attrs:
                raise NotImplementedError('output attr {} is not found in this reader.'.format(i))
            ret[i] = attrs[i]
        return ret

    def load_data(self, input_file, batch_size, num_epochs=None, \
                  file_format='tsv', shuffle_train=True):
        """将磁盘上的数据载入到reader中。

        注意：实现该方法时需要同步创建self._batch_size和self._num_epochs。

        Args:
            input_file: 数据集文件路径。文件格式需要满足`file_format`参数的要求。
            batch_size: 迭代器每次yield出的样本数量。注意：当环境中存在多个GPU时，batch_size需要保证被GPU卡数整除。
            num_epochs: 数据集遍历次数。默认为None, 在单任务模式下代表遍历一次，在多任务模式下该参数会被上层的Trainer进行自动赋值。该参数仅对训练阶段有效。
            file_format: 输入文件的文件格式。目前支持的格式: tsv. 默认为tsv.
            shuffle_train: 是否打乱训练集中的样本。默认为True。该参数仅对训练阶段有效。
        """
        raise NotImplementedError()

    @property
    def outputs_attr(self):
        """描述reader输出对象（被yield出的对象）的属性，包含各个对象的名字、shape以及数据类型。当某个对象为标量数据
        类型（如str, int, float等）时，shape设置为空列表[]，当某个对象的某个维度长度可变时，shape中的相应维度设置为-1。
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
    
    def _iterator(self):
        """数据集遍历接口，注意，当数据集遍历到尾部时该接口应自动完成指针重置，即重新从数据集头部开始新的遍历。
        Yield:
            dict类型。符合outputs_attr描述的当前step的输出对象。
        """
        raise NotImplementedError()

    def get_epoch_outputs(self):
        """返回数据集每个epoch遍历后的输出对象。"""
        raise NotImplementedError()

    @property
    def num_examples(self):
        """数据集中的样本数量，即每个epoch中iterator所生成的样本数。注意，使用滑动窗口等可能导致数据集样本数发生变化的策略时
        该接口应返回runtime阶段的实际样本数。"""
        raise NotImplementedError()

    @property
    def num_epochs(self):
        """数据集遍历次数"""
        return self._num_epochs
```



在基类的基础上，定义一个全新的Reader时需要至少实现的方法有：

- \_\_init\_\_
- outputs_attr
- load_data
- _iterator
- num_examples

可以重写的方法有：

- get_epoch_outputs

