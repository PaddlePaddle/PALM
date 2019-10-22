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
import sys
import random
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import layers


def _check_and_adapt_shape_dtype(rt_val, attr):
    if not isinstance(rt_val, np.ndarray):
        rt_val = np.array(rt_val)
        assert rt_val.dtype != np.dtype('O'), "yielded data is not a valid tensor(number of elements on some dimension may differ)."
        if rt_val.dtype == np.dtype('float64'):
            rt_val = rt_val.astype('float32')
    
    shape, dtype = attr
    assert rt_val.dtype == np.dtype(dtype), "yielded data type not consistent with attr settings."
    assert len(shape) == rt_val.ndim, "yielded data rank(ndim) not consistent with attr settings."
    for rt, exp in zip(rt_val.shape, shape):
        if exp is None or exp < 0:
            continue
        assert rt == exp, "yielded data shape is not consistent with attr settings.\nExpected:{}\nActual:{}".format(exp, rt)
    return rt_val
    

def _zero_batch(attrs):
    pos_attrs = []
    for shape, dtype in attrs:
        pos_shape = [size if size and size > 0 else 1 for size in shape]
        pos_attrs.append([pos_shape, dtype])

    return [np.zeros(shape=shape, dtype=dtype) for shape, dtype in pos_attrs]


def _zero_batch_x(attrs, batch_size):
    pos_attrs = []
    for shape, dtype in attrs:
        # pos_shape = [size if size and size > 0 else 5 for size in shape]
        pos_shape = [size for size in shape]
        if pos_shape[0] == -1:
            pos_shape[0] = batch_size
        if pos_shape[1] == -1:
            pos_shape[1] = 512 # max seq len
        pos_attrs.append([pos_shape, dtype])

    return [np.zeros(shape=shape, dtype=dtype) for shape, dtype in pos_attrs]


def create_net_inputs(input_attrs, async=False, iterator_fn=None, dev_count=1, n_prefetch=1):
    inputs = []
    ret = {}
    for name, shape, dtype in input_attrs:
        p = layers.data(name, shape=shape, dtype=dtype)
        ret[name] = p
        inputs.append(p)

    if async:
        assert iterator_fn is not None, "iterator_fn is needed for building async input layer."
        reader = fluid.io.PyReader(inputs, capacity=dev_count*n_prefetch, iterable=False)
        reader.decorate_batch_generator(iterator_fn)
        reader.start()

    return ret


def create_iterator_fn(iterator, iterator_prefix, shape_and_dtypes, outname_to_pos, verbose=0):

    def iterator():
        v = verbose
        while True:
            results = _zero_batch(shape_and_dtypes)

            outputs = next(iterator) # dict type
            prefix = iterator_prefixe
            for outname, val in outputs.items():
                task_outname = prefix + '/' + outname

                if outname in outname_to_pos:
                    idx = outname_to_pos[outname]
                    val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx])
                    results[idx] = val

                if task_outname in outname_to_pos:
                    idx = outname_to_pos[task_outname]
                    val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx])
                    results[idx] = val

            yield results

    return iterator


def create_joint_iterator_fn(iterators, iterator_prefixes, joint_shape_and_dtypes, mrs, outname_to_pos, dev_count=1, keep_one_task=True, verbose=0, batch_size=None):
    """
        joint_shape_and_dtypes: 本质上是根据bb和parad的attr设定的，并且由reader中的attr自动填充-1（可变）维度得到，因此通过与iterator的校验可以完成runtime的batch正确性检查
    """

    task_ids = range(len(iterators))
    weights = [mr / float(sum(mrs)) for mr in mrs]
    if not keep_one_task:
        dev_count = 1

    # build fake batch
    # 注意这种方法会导致一个问题，用户将某任务的mix ratio设置成0后，并不能避免从该任务上读数据，若用户将数据集删掉则会导致崩溃；不过相比之前的zero batch方法，这种方法不必作出只能有一个size=-1的维度且第0维的-1必须是batch size的假设
    results = _zero_batch(joint_shape_and_dtypes)
    outbuf = {}
    for id in task_ids:
        outputs = next(iterators[id]) # dict type
        outbuf[id] = outputs
        prefix = iterator_prefixes[id]
        for outname, val in outputs.items():
            task_outname = prefix + '/' + outname

            if outname in outname_to_pos:
                idx = outname_to_pos[outname]
                val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx])
                results[idx] = val

            if task_outname in outname_to_pos:
                idx = outname_to_pos[task_outname]
                val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx])
                results[idx] = val

    fake_batch = results
    dev_count_bak = dev_count

    def iterator():
        v = verbose
        while True:
            id = np.random.choice(task_ids, p=weights)
            results = fake_batch
            if v > 0:
                print('----- debug joint iterator -----')
                print('sampled task id: '+str(id))
            task_id_tensor = np.array([[id]]).astype("int64")
            results[0] = task_id_tensor
            
            for i in range(dev_count):
                # results = _zero_batch(joint_shape_and_dtypes, batch_size=batch_size)
                # results[0] = task_id_tensor
                if id in outbuf:
                    outputs = outbuf[id]
                    del outbuf[id]
                else:
                    outputs = next(iterators[id]) # dict type

                prefix = iterator_prefixes[id]
                for outname, val in outputs.items():
                    if v > 0:
                        print('reader generate: '+outname)
                    task_outname = prefix + '/' + outname

                    if outname in outname_to_pos:
                        idx = outname_to_pos[outname]
                        if v > 0:
                            print(outname + ' is insert in idx ' + str(idx))
                        val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx])
                        results[idx] = val

                    if task_outname in outname_to_pos:
                        idx = outname_to_pos[task_outname]
                        if v > 0:
                            print(task_outname + ' is insert in idx ' + str(idx))
                        val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx])
                        results[idx] = val

                if v > 0:
                    print('yielded batch len and shapes:')
                    print(len(results))
                    for i in results:
                        print(np.shape(i))
                    print('')
                    v -= 1
                yield results

    return iterator


def merge_input_attrs(backbone_attr, task_attrs, insert_taskid=True):
    """
    Args:
        task_attrs(list[dict]|dict): task input attributes, key=attr_name, val=[shape, dtype], support single task and nested tasks
    """
    if isinstance(task_attrs, dict):
        task_attrs = [task_attrs]

    if insert_taskid:
        ret = [([1,1], 'int64')]
        names = ['__task_id']
        start = 1
    else:
        ret = []
        names = []
        start = 0
        
    names += sorted(backbone_attr.keys())
    ret.extend([backbone_attr[k] for k in names[start:]])
    name_to_position = {}
    # pos=0 is for task_id, thus we start from 1
    for pos, k in enumerate(names):
        name_to_position[k] = pos
    for task_attr in task_attrs:
        task_names = sorted(task_attr.keys())
        names.extend(task_names)
        ret.extend([task_attr[k] for k in task_names])
        for pos, k in enumerate(task_names, start=len(name_to_position)):
            name_to_position[k] = pos
    return names, ret, name_to_position
    

