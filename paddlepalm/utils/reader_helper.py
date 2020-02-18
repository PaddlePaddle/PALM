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
import logging
import numpy as np
import paddle
from paddle import fluid
from paddle.fluid import layers
from paddlepalm.distribute import gpu_dev_count, cpu_dev_count
import six
dev_count = 1 if gpu_dev_count <= 1 else gpu_dev_count


def create_feed_batch_process_fn(net_inputs):
    
    def feed_batch_process_fn(data, id=-1, phase='train', is_multi=False):
        temp = {}
        if dev_count > 1 and phase=='train' and is_multi:
            inputs = net_inputs[id]
        else:
            inputs= net_inputs

        for q, var in inputs.items():
            
            if isinstance(var, str) or (six.PY3 and isinstance(var, bytes)) or (six.PY2 and isinstance(var, unicode)):
                temp[var] = data[q]
            else:
                temp[var.name] = data[q]
        return temp

    return feed_batch_process_fn


# def create_multihead_feed_batch_process_fn(net_inputs):
# 
#     def feed_batch_process_fn(data, id=-1):
#         # temps = {}
#         # for i in range(len(net_inputs)):
#         temp = {}
#         inputs = net_inputs[id] if id != -1 else net_inputs
#         
#         for q, var in inputs.items():
#             if isinstance(var, str) or isinstance(var, unicode):
#                 temp[var] = data[q]
#             else:
#                 temp[var.name] = data[q]
#             # temps[i] = temp
#             
#         return temp
# 
#     return feed_batch_process_fn


def check_io(in_attr, out_attr, strict=False, in_name="left", out_name="right"):
    for name, attr in in_attr.items():
        assert name in out_attr, in_name+': '+name+' not found in '+out_name
        if attr != out_attr[name]:
            if strict:
                raise ValueError(name+': shape or dtype not consistent!')
            else:
                logging.warning('{}: shape or dtype not consistent!\n{}:\n{}\n{}:\n{}'.format(name, in_name, attr, out_name, out_attr[name]))


def _check_and_adapt_shape_dtype(rt_val, attr, message=""):
    if not isinstance(rt_val, np.ndarray):
        if rt_val is None:
            raise Exception(message+": get None value. ")
        rt_val = np.array(rt_val)
        assert rt_val.dtype != np.dtype('O'), message+"yielded data is not a valid tensor (number of elements on some dimension may not consistent): {}".format(rt_val)
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


def create_net_inputs(input_attrs, is_async=False, iterator_fn=None, dev_count=1, n_prefetch=1):
    inputs = []
    ret = {}
    for name, shape, dtype in input_attrs:
        p = layers.data(name, shape=shape, dtype=dtype)
        ret[name] = p
        inputs.append(p)

    if is_async:
        assert iterator_fn is not None, "iterator_fn is needed for building async input layer."
        reader = fluid.io.PyReader(inputs, capacity=dev_count, iterable=False)
        reader.decorate_batch_generator(iterator_fn)
        reader.start()

    return ret


def create_iterator_fn(iterator, iterator_prefix, shape_and_dtypes, outname_to_pos, verbose=0, return_type='list'):

    pos_to_outname = {j:i for i,j in outname_to_pos.items()}
    
    def iterator_fn():
        v = verbose
        for outputs in iterator:
            results = [None] * len(outname_to_pos)
            prefix = iterator_prefix
            for outname, val in outputs.items():
                task_outname = prefix + '.' + outname

                if outname in outname_to_pos:
                    idx = outname_to_pos[outname]
                    val = _check_and_adapt_shape_dtype(val, shape_and_dtypes[idx])
                    results[idx] = val

                if task_outname in outname_to_pos:
                    idx = outname_to_pos[task_outname]
                    val = _check_and_adapt_shape_dtype(val, shape_and_dtypes[idx])
                    results[idx] = val
            if return_type == 'list':
                yield results
            elif return_type == 'dict':
                temp = {}
                for pos, i in enumerate(results):
                    temp[pos_to_outname[pos]] = i

                yield temp

    return iterator_fn

def create_multihead_iterator_fn(iterators, iterator_prefixes, joint_shape_and_dtypes, mrs, names, outname_to_pos, dev_count=1, keep_one_task=True):
    task_ids = range(len(iterators))
    weights = [mr / float(sum(mrs)) for mr in mrs]
    if not keep_one_task:
        dev_count = 1

    def iterator():
        while True:
            id = np.random.choice(task_ids, p=weights)
            task_id_tensor = np.array([id]).astype("int64")
            
            for i in range(dev_count):
                
                outputs = next(iterators[id]) # dict type

                prefix = iterator_prefixes[id]
                results = {}
                results['__task_id'] = task_id_tensor
                for outname, val in outputs.items():
                    task_outname = prefix + '.' + outname

                    if outname in names[id]:
                        idx = outname_to_pos[id][outname]
                        val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[id][idx], message=outname+': ')
                        results[outname] = val

                    if task_outname in names[id]:
                        idx = outname_to_pos[id][task_outname]
                        val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[id][idx], message=task_outname+': ')
                        results[task_outname] = val

                yield results

    return iterator


def create_joint_iterator_fn(iterators, iterator_prefixes, joint_shape_and_dtypes, mrs, outname_to_pos, dev_count=1, keep_one_task=True, verbose=0):
    """
        joint_shape_and_dtypes: 本质上是根据bb和parad的attr设定的，并且由reader中的attr自动填充-1（可变）维度得到，因此通过与iterator的校验可以完成runtime的batch正确性检查
    """

    task_ids = range(len(iterators))
    weights = [mr / float(sum(mrs)) for mr in mrs]
    if not keep_one_task:
        dev_count = 1

    results = _zero_batch(joint_shape_and_dtypes)
    outbuf = {}
    for id in task_ids:
        outputs = next(iterators[id]) # dict type
        outbuf[id] = outputs
        prefix = iterator_prefixes[id]
        for outname, val in outputs.items():
            task_outname = prefix + '.' + outname

            if outname in outname_to_pos:
                idx = outname_to_pos[outname]
                val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx], message=outname+': ')
                results[idx] = val

            if task_outname in outname_to_pos:
                idx = outname_to_pos[task_outname]
                val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx], message=task_outname+': ')
                results[idx] = val

    fake_batch = results
    dev_count_bak = dev_count

    def iterator():
        v = verbose
        has_show_warn = False
        while True:
            id = np.random.choice(task_ids, p=weights)
            results = fake_batch
            if v > 0:
                print('----- debug joint iterator -----')
                print('sampled task id: '+str(id))
            task_id_tensor = np.array([[id]]).astype("int64")
            
            for i in range(dev_count):
                
                results[outname_to_pos['__task_id']] = task_id_tensor
                assert outname_to_pos['__task_id'] == 0

                if id in outbuf:
                    outputs = outbuf[id]
                    del outbuf[id]
                else:
                    outputs = next(iterators[id]) # dict type

                if 'token_ids' in outputs:
                    val1 = len(outputs['token_ids'])
                    val = _check_and_adapt_shape_dtype([val1], [[1], 'int64'])
                    results[outname_to_pos['batch_size']] = val

                    val2 = len(outputs['token_ids'][0])
                    val = _check_and_adapt_shape_dtype([val2], [[1], 'int64'])
                    results[outname_to_pos['seqlen']] = val

                    val = _check_and_adapt_shape_dtype([val1*val2], [[1], 'int64'])
                    results[outname_to_pos['batchsize_x_seqlen']] = val
                else:
                    if not has_show_warn:
                        print('WARNING: token_ids not found in current batch, failed to yield batch_size, seqlen and batchsize_x_seqlen. (This message would be shown only once.)')
                        has_show_warn = True

                prefix = iterator_prefixes[id]
                for outname, val in outputs.items():
                    if v > 0:
                        print('reader generate: '+outname)
                    task_outname = prefix + '.' + outname

                    if outname in outname_to_pos:
                        idx = outname_to_pos[outname]
                        if v > 0:
                            print(outname + ' is insert in idx ' + str(idx))
                        val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx], message=outname+': ')
                        results[idx] = val

                    if task_outname in outname_to_pos:
                        idx = outname_to_pos[task_outname]
                        if v > 0:
                            print(task_outname + ' is insert in idx ' + str(idx))
                        val = _check_and_adapt_shape_dtype(val, joint_shape_and_dtypes[idx], message=task_outname+': ')
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


def merge_input_attrs(backbone_attr, task_attrs, insert_taskid=True, insert_batchsize=False, insert_seqlen=False, insert_batchsize_x_seqlen=False):
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
