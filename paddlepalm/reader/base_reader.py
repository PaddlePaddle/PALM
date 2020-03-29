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
