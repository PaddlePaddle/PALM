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


def _check_and_adapt_shape_dtype(rt_val, attr, message=""):
    if not isinstance(rt_val, np.ndarray):
        
        rt_val = np.array(rt_val)
        assert rt_val.dtype != np.dtype('O'), "yielded data is not a valid tensor(number of elements on some dimension may differ)."
        if rt_val.dtype == np.dtype('float64'):
            rt_val = rt_val.astype('float32')
    
    shape, dtype = attr
   
    assert rt_val.dtype == np.dtype(dtype), message+"yielded data type not consistent with attr settings. Expect: {}, receive: {}.".format(rt_val.dtype, np.dtype(dtype))
    assert len(shape) == rt_val.ndim, message+"yielded data rank(ndim) not consistent with attr settings. Expect: {}, receive: {}.".format(len(shape), rt_val.ndim)
    for rt, exp in zip(rt_val.shape, shape):
        if exp is None or exp < 0:
            continue
        assert rt == exp, "yielded data shape is not consistent with attr settings.Expected:{}Actual:{}".format(exp, rt)
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
        p = fluid.data(name, shape=shape, dtype=dtype)
        ret[name] = p
        inputs.append(p)

    if async:
        assert iterator_fn is not None, "iterator_fn is needed for building async input layer."
        reader = fluid.io.PyReader(inputs, capacity=dev_count, iterable=False)
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


def create_joint_iterator_fn(iterators, iterator_prefixes, joint_shape_and_dtypes, mrs, outname_to_pos, dev_count=1, keep_one_task=True, verbose=0, return_type='list'):
    """
        joint_shape_and_dtypes: 本质上是根据bb和parad的attr设定的，并且由reader中的attr自动填充-1（可变）维度得到，因此通过与iterator的校验可以完成runtime的batch正确性检查
    """

    task_ids = range(len(iterators))
    weights = [mr / float(sum(mrs)) for mr in mrs]
    if not keep_one_task:
        dev_count = 1

    results = {}
    pos_to_outname = {}
    for id in task_ids:
        pos_to_outname[id] = {j:i for i,j in outname_to_pos[id].items()}
        result = _zero_batch(joint_shape_and_dtypes[id])
        outbuf = {}
        outputs = next(iterators[id]) # dict type
        outbuf[id] = outputs
        prefix = iterator_prefixes[id]
        for outname, val in outputs.items():
            task_outname = prefix + '/' + outname

            if outname in outname_to_pos[id]:
                idx = outname_to_pos[id][outname]
                val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[id][idx], message=outname+': ')
                result[idx] = val

            if task_outname in outname_to_pos[id]:
                idx = outname_to_pos[id][task_outname]
                val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[id][idx], message=task_outname+': ')
                result[idx] = val
        results[id] = result

    def iterator():
        v = verbose
        has_show_warn = False
        while True:
            id = np.random.choice(task_ids, p=weights)
            if v > 0:
                print('----- debug joint iterator -----')
                print('sampled task id: '+str(id))
            task_id_tensor = np.array([[id]]).astype("int64")
            
            for i in range(dev_count):
                
                results[id][outname_to_pos[id]['__task_id']] = task_id_tensor
                assert outname_to_pos[id]['__task_id'] == 0

                if id in outbuf:
                    outputs = outbuf[id]
                    del outbuf[id]
                else:
                    outputs = next(iterators[id]) # dict type

                if 'token_ids' in outputs:
                    val1 = len(outputs['token_ids'])
                    val = _check_and_adapt_shape_dtype(np.array([val1], dtype='int64'), [[1], 'int64'], iterator_prefixes[id]+' tokenids: ')
                    results[id][outname_to_pos[id]['batch_size']] = val

                    val2 = len(outputs['token_ids'][0])
                    val = _check_and_adapt_shape_dtype(np.array([val2], dtype='int64'), [[1], 'int64'])
                    results[id][outname_to_pos[id]['seqlen']] = val

                    val = _check_and_adapt_shape_dtype(np.array([val1*val2], dtype='int64'), [[1], 'int64'])
                    results[id][outname_to_pos[id]['batchsize_x_seqlen']] = val
                else:
                    if not has_show_warn:
                        print('WARNING: token_ids not found in current batch, failed to yield batch_size, seqlen and batchsize_x_seqlen. (This message would be shown only once.)')
                        has_show_warn = True

                prefix = iterator_prefixes[id]
                for outname, val in outputs.items():
                    if v > 0:
                        print('reader generate: '+outname)
                    task_outname = prefix + '/' + outname

                    if outname in outname_to_pos[id]:
                        idx = outname_to_pos[id][outname]
                        if v > 0:
                            print(outname + ' is insert in idx ' + str(idx))
                        val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[id][idx], message=outname+': ')
                        results[id][idx] = val

                    if task_outname in outname_to_pos[id]:
                        idx = outname_to_pos[id][task_outname]
                        if v > 0:
                            print(task_outname + ' is insert in idx ' + str(idx))
                        val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[id][idx], message=task_outname+': ')
                        results[id][idx] = val

                if v > 0:
                    print('yielded batch len and shapes:')
                    print(len(results[id]))
                    for i in results[id]:
                        print(np.shape(i))
                    print('')
                    v -= 1
                if return_type == 'list':
                    yield results[id]
                elif return_type == 'dict':
                    temp = {}
                    for pos, i in enumerate(results[id]):
                        temp[pos_to_outname[id][pos]] = i
                    yield temp

    return iterator


def merge_input_attrs(backbone_attr, task_attrs, insert_taskid=True, insert_batchsize=True, insert_seqlen=True, insert_batchsize_x_seqlen=True):
    """
    Args:
        task_attrs(list[dict]|dict): task input attributes, key=attr_name, val=[shape, dtype], support single task and nested tasks
    """
    if isinstance(task_attrs, dict):
        task_attrs = [task_attrs]

    ret = []
    names = []
    start = 0
    if insert_taskid:
        ret.append(([1, 1], 'int64'))
        names.append('__task_id')
        start += 1
    
    if insert_batchsize:
        ret.append(([1], 'int64'))
        names.append('batch_size')
        start += 1

    if insert_seqlen:
        ret.append(([1], 'int64'))
        names.append('seqlen')
        start += 1

    if insert_batchsize_x_seqlen:
        ret.append(([1], 'int64'))
        names.append(u'batchsize_x_seqlen')
        start += 1
        
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
    

