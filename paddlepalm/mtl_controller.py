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

from __future__ import print_function

import os
import sys
import importlib
import multiprocessing
from paddle import fluid
from paddle.fluid import layers
import yaml
import json
import logging
import time
import numpy as np

from paddlepalm.utils.saver import init_pretraining_params, init_checkpoint
from paddlepalm.utils.config_helper import PDConfig
from paddlepalm.utils.print_helper import print_dict
from paddlepalm.utils.reader_helper import create_net_inputs, create_iterator_fn, create_joint_iterator_fn, merge_input_attrs 
from paddlepalm.distribute import data_feeder, decode_fake

from default_settings import *
from task_instance import TaskInstance, check_instances


DEBUG=False
VERBOSE=0

def _get_basename(f):
    return os.path.splitext(f)[0]


def _get_suffix(f):
    return os.path.splitext(f)[-1]


def _parse_yaml(f, asdict=True, support_cmd_line=False):
    assert os.path.exists(f), "file {} not found.".format(f)
    if support_cmd_line:
        args = PDConfig(yaml_file=f, fuse_args=True)
        args.build()
        return args.asdict() if asdict else args
    else:
        if asdict:
            with open(f, "r") as fin: 
                yaml_config = yaml.load(fin, Loader=yaml.SafeLoader)
            return yaml_config
        else:
            raise NotImplementedError()


def _parse_json(f, asdict=True, support_cmd_line=False):
    assert os.path.exists(f), "file {} not found.".format(f)
    if support_cmd_line:
        args = PDConfig(json_file=f, fuse_args=support_cmd_line)
        args.build()
        return args.asdict() if asdict else args
    else:
        if asdict:
            with open(f, "r") as fin: 
                config = json.load(fin)
            return config
        else:
            raise NotImplementedError()
            

def _parse_list(string, astype=str):
    assert isinstance(string, str), "{} is not a string.".format(string)
    if ',' not in string:
        return [astype(string)]
    string = string.replace(',', ' ')
    return [astype(i) for i in string.split()]


def _try_float(s):
    try:
        float(s)
        return(float(s))
    except:
        return s


def _check_conf(conf, checklist=None):
    assert isinstance(conf, dict), "{} is not a dict.".format(conf)
    ret = {}
    for k,v in conf.items():
        if isinstance(v, str):
            v = _try_float(v)
        ret[k] = v
    if checklist is not None:
        for k, t in checklist:
            assert k in ret, "required argument {} is NOT exist in config file.".format(k)
            assert isintance(ret[k], t), "value type of argument {} should be {}".format(k, t)
    return ret


# TODO: 增加None机制，允许hidden size、batch size和seqlen设置为None
def _check_io(in_attr, out_attr, strict=False, in_name="left", out_name="right"):
    for name, attr in in_attr.items():
        assert name in out_attr, in_name+': '+name+' not found in '+out_name
        if attr != out_attr[name]:
            if strict:
                raise ValueError(name+': shape or dtype not consistent!')
            else:
                logging.warning('{}: shape or dtype not consistent!\n{}:\n{}\n{}:\n{}'.format(name, in_name, attr, out_name, out_attr[name]))


def _merge_conf(conf1, conf2, conf1_first=True, strict=False):
    assert isinstance(conf1, dict), "{} is not a dict.".format(conf1)
    assert isinstance(conf2, dict), "{} is not a dict.".format(conf2)
    base_conf = conf2 if conf1_first else conf1
    base_conf = base_conf.copy()
    new_conf = conf1 if conf1_first else conf2

    for k, v in new_conf.items():
        if k in base_conf:
            if base_conf[k] != v:
                raise Warning("value of argument {} has been updated to {}.".format(k, v))
        else:
            if strict:
                continue
            
        base_conf[k] = v
    return base_conf


def _encode_inputs(inputs, scope_name, sep='/', cand_set=None):
    outputs = {}
    for k, v in inputs.items():
        if cand_set is not None:
            if k in cand_set:
                outputs[k] = v
            if scope_name+sep+k in cand_set:
                outputs[scope_name+sep+k] = v
        else:
            outputs[scope_name+sep+k] = v
    return outputs


def _decode_inputs(inputs, scope_name, sep='/', keep_unk_keys=True):
    outputs = {}
    for name, value in inputs.items():
        # var for backbone are also available to tasks
        if keep_unk_keys and sep not in name:
            outputs[name] = value
        # var for this inst
        if name.startswith(scope_name+'/'):
            outputs[name[len(scope_name+'/'):]] = value
    return outputs


def _init_env(use_gpu):
    if use_gpu:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    return fluid.Executor(place), dev_count


def _fit_attr(conf, fit_attr, strict=False):
    for i, attr in fit_attr.items():
        if i not in conf:
            if strict:
                raise Exception('Argument {} is required to create a controller.'.format(i))
            else:
                continue
        conf[i] = attr(conf[i])
    return conf


def create_feed_batch_process_fn(net_inputs):


    def feed_batch_process_fn(data, id=-1):
        # temps = {}
        # for i in range(len(net_inputs)):
        temp = {}
        inputs = net_inputs[id] if id != -1 else net_inputs
        
        for q, var in inputs.items():
            if isinstance(var, str) or isinstance(var, unicode):
                temp[var] = data[q]
            else:
                temp[var.name] = data[q]
            # temps[i] = temp
            
        return temp

    return feed_batch_process_fn


class Controller(object):

    def __init__(self, config, task_dir='.', for_train=True):
        """
        Args:
            config: (str|dict) 字符串类型时，给出yaml格式的config配置文件路径；
        """

        self._for_train = for_train
        assert isinstance(config, str) or isinstance(config, dict), "a config dict or config file path is required to create a Controller."

        if isinstance(config, str):
            mtl_conf = _parse_yaml(config, support_cmd_line=True)
        else:
            mtl_conf = config
                
        mtl_conf = _check_conf(mtl_conf)
        mtl_conf = _fit_attr(mtl_conf, REQUIRED_ARGS, strict=True)
        mtl_conf = _fit_attr(mtl_conf, OPTIONAL_ARGS, strict=False)

        exe, dev_count = _init_env(use_gpu=mtl_conf.get('use_gpu', True))
        self.exe = exe
        self.dev_count = dev_count
        self.batch_size = mtl_conf.get('batch_size')

        print_dict(mtl_conf, title='global configuration')

        # parse task instances and target tags
        instnames = _parse_list(mtl_conf['task_instance'])
        assert len(instnames) == len(set(instnames)), "repeated task_instance is NOT supported."
        num_instances = len(instnames)
        self.num_instances = num_instances

        instname_to_conf = {}
        instname_to_id = {}
        for id, instname in enumerate(instnames):
            instpath = os.path.join(task_dir, instname+'.yaml')
            conf = _parse_yaml(instpath, support_cmd_line=False)
            # conf = _check_conf(conf, TASK_INSTANCE_REQUIRED_ARGS)
            conf = _check_conf(conf)
            temp_conf = _merge_conf(mtl_conf, conf, strict=True)
            print_dict(temp_conf, title='{} configuration'.format(instname))
            conf = _merge_conf(mtl_conf, conf)
            
            instname_to_conf[instname] = conf
            instname_to_id[instname] = id

        # prepare backbone
        if 'backbone_config_path' in mtl_conf:
            bb_conf = _parse_json(mtl_conf['backbone_config_path'])
            bb_conf = _merge_conf(mtl_conf, bb_conf)
        else:
            bb_conf = mtl_conf
        print_dict(bb_conf, title = 'backbone configuration'.format(instname))

        bb_name = mtl_conf['backbone']
        bb_mod = importlib.import_module(BACKBONE_DIR + '.' + bb_name)
        Backbone = getattr(bb_mod, 'Model')

        # create task instances
        instances = []
        for name in instnames:
            instances.append(TaskInstance(name, instname_to_id[name], instname_to_conf[name]))

        check_instances(instances)

        # parse target_tag
        if 'target_tag' in mtl_conf:
            target_tag = str(mtl_conf['target_tag'])
            tags = _parse_list(target_tag, astype=int)
            assert len(tags) == len(instnames), "number of target_tag is NOT consistent with that in task_instance."
            for tag, inst in zip(tags, instances):
                inst.is_target = tag
        else:
            tags = [i.is_target for i in instances]
        num_targets = sum(tags)
        num_auxes = num_instances - num_targets

        # parse mix ratios
        if 'mix_ratio' in mtl_conf:
            mix_ratio = str(mtl_conf['mix_ratio'])
            mrs = _parse_list(mix_ratio, astype=float)
            assert len(mrs) == num_instances, "number of mix_ratios is NOT consistent with num_instances."
        else:
            mrs = [1.0] * num_instances

        for mr, inst in zip(mrs, instances):
            inst.mix_ratio = mr

        # parse task layer reuse tags
        instname_to_reusehost = {i:i for i in instnames}
        if 'task_reuse_tag' in mtl_conf:
            tags = _parse_list(mtl_conf['task_reuse_tag'], astype=int)
            assert len(tags) == num_targets, 'number of reuse_tags is NOT consistent with number of instances.'
        else:
            tags = []
            mapper = {}
            for inst in instances:
                history = set()
                history.add(inst.name)
                cur_inst = inst
                while True:
                    if cur_inst.task_reuse_scope in history:
                        mapper[inst.name] = len(tags)
                        break
                    elif cur_inst.task_reuse_scope in mapper:
                        mapper[inst.name] = mapper[cur_inst.task_reuse_scope]
                        break
                    else:
                        cur_inst = name_to_instance[cur_inst.task_reuse_scope]
                        history.add(cur_inst.name)

                tags.append(mapper[inst.name])

        for i in range(1, num_instances):
            for j in range(i):
                if tags[i] == tags[j]:
                    assert instances[i].Paradigm == \
                            instances[j].Paradigm, \
                            "paradigm of reuse tasks should be consistent"
                    instances[i].task_reuse_scope = instances[j].name
                    break

        self.instances = instances
        self.mrs = mrs
        self.Backbone = Backbone
        self.bb_conf = bb_conf
        self.bb_name = bb_name

        self.has_init_train = False
        self.has_init_pred = False

        if self._for_train:
            print("initialing for training...")
            self._init_train()
            self.has_init_train = True
            
    def _init_train(self):
        
        instances = self.instances
        Backbone = self.Backbone
        bb_conf = self.bb_conf
        bb_name = self.bb_name
        dev_count = self.dev_count
        num_instances = len(instances)
        mrs = self.mrs
        branch = fluid.data(name="branch",shape=[1],dtype='int64')

        # set first_target/main task instance
        main_inst = None
        for inst in instances:
            if inst.is_target:
                main_inst = inst
                inst.is_first_target = True
                break
        main_conf = main_inst.config
        if not os.path.exists(main_conf['save_path']):
            os.makedirs(main_conf['save_path'])
            os.makedirs(os.path.join(main_conf['save_path'], 'ckpt'))
        
        # prepare backbone
        train_backbone = Backbone(bb_conf, phase='train')
        pred_backbone = Backbone(bb_conf, phase='pred')

        # create reader, task
        # then check i/o across reader, backbone and task_layer
        
        # check_fns = {}
        task_attrs = {}
        pred_task_attrs = []
        joint_input_names = {}
        joint_shape_and_dtypes = {}
        name_to_position = {}
        for i in range(num_instances):
            # def check_tasks():
            #     i = s 
            #     def checkeach():
                    
            train_reader = instances[i].Reader(instances[i].config, phase='train')
            instances[i].reader['train'] = train_reader
            train_parad = instances[i].Paradigm(instances[i].config, phase='train', backbone_config=bb_conf)
            instances[i].task_layer['train'] = train_parad
            task_attr_from_reader = _encode_inputs(train_parad.inputs_attrs['reader'], instances[i].name)
            task_attrs[i] = task_attr_from_reader

            _check_io(train_backbone.inputs_attr, train_reader.outputs_attr, in_name=bb_name+'_backbone', out_name='reader.train')
            _check_io(train_parad.inputs_attrs['reader'], train_reader.outputs_attr, in_name='task_paradigm.train.reader', out_name='reader.train')
            _check_io(train_parad.inputs_attrs['backbone'], train_backbone.outputs_attr, in_name='task_paradigm.train.backbone', out_name=bb_name+'_backbone')
            # merge reader input attrs from backbone and task_instances
            # pred_joint_input_names = []
            # pred_joint_shape_and_dtypes = []
            if instances[i].is_target:
                if 'pred_file' not in instances[i].config:
                    instances[i].config['pred_file'] = ''
                pred_reader = instances[i].Reader(instances[i].config, phase='pred')
                pred_parad = instances[i].Paradigm(instances[i].config, phase='pred', backbone_config=bb_conf)
                instances[i].task_layer['pred'] = pred_parad
                task_attr_from_reader = _encode_inputs(pred_parad.inputs_attrs['reader'], instances[i].name)
                pred_task_attrs.append(task_attr_from_reader)
                _check_io(pred_backbone.inputs_attr, pred_reader.outputs_attr, in_name=bb_name+'_backbone', out_name='reader.pred')
                _check_io(pred_parad.inputs_attrs['reader'], pred_reader.outputs_attr, in_name='task_paradigm.pred.reader', out_name='reader.pred')
                _check_io(pred_parad.inputs_attrs['backbone'], pred_backbone.outputs_attr, in_name='task_paradigm.pred.backbone', out_name=bb_name+'_backbone')
                # pred_joint_input_names, pred_joint_shape_and_dtypes, _ = merge_input_attrs(pred_backbone.inputs_attr, pred_task_attrs, insert_taskid=False, insert_batchsize=False, insert_seqlen=False, insert_batchsize_x_seqlen=False)
                #     return joint_input_names[i], joint_shape_and_dtypes[i], name_to_position[i], pred_joint_input_names, pred_joint_shape_and_dtypes
                #   return checkeach
                # check_fns[i] = check_tasks()
            joint_input_names[i], joint_shape_and_dtypes[i], name_to_position[i] = merge_input_attrs(train_backbone.inputs_attr, task_attrs[i])
           
        pred_joint_input_names, pred_joint_shape_and_dtypes, _ = merge_input_attrs(pred_backbone.inputs_attr, pred_task_attrs, insert_taskid=False, insert_batchsize=False, insert_seqlen=False, insert_batchsize_x_seqlen=False)
      
            
        # shapes: [task_id, shapes_of_backbone, shapes_of_inst1, ..., shapes_of_instN]

        if DEBUG:
            print('----- for debug -----')
            print('joint input names:')
            print(joint_input_names)
            print('joint input shape and dtypes:')
            print(joint_shape_and_dtypes)

        # load data 
        data_fns={}
        for i in range(num_instances):
            print(instances[i].name+": preparing data...", end='')
            instances[i].reader['train'].load_data()
            print('ok!')

        # merge dataset iterators and create net input vars
        iterators = []
        prefixes = []
        mrs = []

        for inst in instances:
            iterators.append(inst.reader['train'].iterator())
            prefixes.append(inst.name)
            mrs.append(inst.mix_ratio)

        joint_iterator_fn = create_joint_iterator_fn(iterators, prefixes, joint_shape_and_dtypes, mrs, name_to_position, dev_count=dev_count, verbose=VERBOSE, return_type='dict')
        self._joint_iterator_fn = joint_iterator_fn

        input_attrs = {}
        net_inputs = {}
        bb_output_vars = {}
        bb_output_fns = {}

        # prepare predict vars for saving inference model
        pred_input_attrs = [[i, j, k] for i, (j,k) in zip(pred_joint_input_names, pred_joint_shape_and_dtypes)]
        pred_prog = fluid.Program()
        pred_init_prog = fluid.Program()
        self._pred_prog = pred_prog

        with fluid.program_guard(main_program = pred_prog, startup_program = pred_init_prog):
           pred_net_inputs = create_net_inputs(pred_input_attrs)
           pred_bb_output_vars = pred_backbone.build(pred_net_inputs, scope_name='__paddlepalm_')

        task_inputs = {}
        task_output_vars = {}
        task_fns = {}

        def get_loss(i):
            input_attrs[i] = [[m, j, k] for m, (j,k) in zip(joint_input_names[i], joint_shape_and_dtypes[i])]
            net_inputs[i] = create_net_inputs(input_attrs[i], async=False)
            # net_inputs = create_net_inputs(input_attrs, async=True, iterator_fn=joint_iterator_fn, dev_count=dev_count, n_prefetch=3)
            bb_output_vars[i] = train_backbone.build(net_inputs[i], scope_name='__paddlepalm_')
            assert sorted(bb_output_vars[i].keys()) == sorted(train_backbone.outputs_attr.keys())

            # build backbone and task layers
            task_inputs[i] = {'backbone': bb_output_vars[i]}
            task_inputs_from_reader = _decode_inputs(net_inputs[i], instances[i].name)
            task_inputs[i]['reader'] = task_inputs_from_reader
        
            scope = instances[i].task_reuse_scope + '/'
            with fluid.unique_name.guard(scope):
                output_vars = instances[i].build_task_layer(task_inputs[i], phase='train', scope=scope)
                output_vars = {instances[i].name+'/'+key: val for key, val in output_vars.items()}
                loss_var = output_vars[instances[i].name+'/loss']
                task_output_vars[i] = output_vars

            if instances[i].is_target:
                with fluid.program_guard(pred_prog, pred_init_prog):
                    cur_inputs = _decode_inputs(pred_net_inputs, instances[i].name)
                    instances[i].pred_input = cur_inputs
                    pred_task_inputs = {'backbone': pred_bb_output_vars, 'reader': cur_inputs}
                    scope = instances[i].task_reuse_scope + '/'
                    with fluid.unique_name.guard(scope):
                        instances[i].build_task_layer(pred_task_inputs, phase='pred', scope=scope)
            return loss_var

        for i in range(num_instances):
            def task_loss():
                task_id = i
                return lambda: get_loss(task_id)
            task_fns[i] = task_loss()

        loss = layers.switch_case(
            branch_index=branch,
            branch_fns=task_fns
        )
        self._switched_loss = loss.name
        main_reader = main_inst.reader['train']

        num_examples = main_reader.num_examples
        for inst in instances:
            max_train_steps = int(main_conf['num_epochs']* inst.mix_ratio * (num_examples // main_conf['batch_size']  // dev_count))
            if inst.is_target:
                print('{}: expected train steps {}.'.format(inst.name, max_train_steps))
            inst.steps_pur_epoch = inst.reader['train'].num_examples // main_conf['batch_size']  // dev_count
            inst.expected_train_steps = max_train_steps

        global_max_train_steps = int(main_conf['num_epochs'] * sum(mrs) * (num_examples // main_conf['batch_size']  // dev_count))
        print('Estimated overall train steps {}.'.format(global_max_train_steps))

        if 'warmup_proportion' in main_conf and main_conf['warmup_proportion'] > 0:
            warmup_steps = int(global_max_train_steps * main_conf['warmup_proportion'])
            print('Warmup steps: '+str(warmup_steps))
        else:
            warmup_steps = 0

        # build optimizer
        if 'optimizer' in main_conf:
            optim_mod = importlib.import_module(OPTIMIZER_DIR + '.' + main_conf['optimizer'])
            optimize = getattr(optim_mod, OPTIMIZE_METHOD)
            optimize(loss, main_conf, max_train_steps, warmup_steps, fluid.default_main_program())

            loss.persistable = True
            if main_conf.get('use_ema', False):
                assert 'ema_decay' in main_conf, "ema_decay should be set when use_ema is enabled."
                ema = fluid.optimizer.ExponentialMovingAverage(main_conf['ema_decay'])
                ema.update()

        # prepare for train
        self.train_backbone = train_backbone
        self.train_program = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(loss_name=loss.name)
        self.saver_program = fluid.default_main_program()

        self.main_inst = main_inst
        self.has_init_train = True
        self.has_init_pred = True
        self._net_inputs = net_inputs

        self.exe.run(fluid.default_startup_program())
        print("\nRandomly initialize parameters...\n")

    def _init_pred(self, instance, infer_model_path):
        inst = instance
        if 'pred_output_path' not in inst.config:
            inst.config['pred_output_path'] = os.path.join(inst.config.get('save_path', '.'), inst.name)

        if not os.path.exists(inst.config['pred_output_path']):
            os.makedirs(inst.config['pred_output_path'])

        pred_backbone = self.Backbone(self.bb_conf, phase='pred')
        pred_parad = inst.Paradigm(inst.config, phase='pred', backbone_config=self.bb_conf)
        inst.task_layer['pred'] = pred_parad
        pred_joint_input_names, pred_joint_shape_and_dtypes, name_to_position = merge_input_attrs(
            pred_backbone.inputs_attr, inst.task_layer['pred'].inputs_attrs['reader'], 
            insert_taskid=False, insert_batchsize=False, insert_seqlen=False, insert_batchsize_x_seqlen=False)

        pred_prog = inst.load(infer_model_path)
        pred_prog = fluid.CompiledProgram(pred_prog).with_data_parallel()
        if inst.reader['pred'] is None:
            pred_reader = inst.Reader(inst.config, phase='pred')
            inst.reader['pred'] = pred_reader
        return pred_prog

    def load_pretrain(self, pretrain_path=None):
        # load pretrain model (or ckpt)
        if pretrain_path is None:
            assert 'pretrain_path' in self.main_conf, "pretrain_path NOT set."
            pretrain_path = self.main_conf['pretrain_path']

        init_pretraining_params(
            self.exe,
            pretrain_path,
            main_program=fluid.default_startup_program())


    def train(self):

        if not self.has_init_train:
            self._init_train()
            self.has_init_train = True

        instances = self.instances
        num_instances = self.num_instances
        main_inst = self.main_inst
        main_conf = main_inst.config

        backbone = self.train_backbone
        train_program = self.train_program
        saver_program = self.saver_program
        finish = []
        for inst in instances:
            if inst.is_target:
                if inst.expected_train_steps > 0:
                    finish.append(False)
                else:
                    finish.append(True)
                    print(inst.name+': train finished!')
                    inst.save()
        
        def train_finish():
            for inst in instances:
                if inst.is_target:
                    if not inst.train_finish:
                        return False
            return True


        # do training
        fetch_names = {}
        fetch_list = []
        main_step = 0 # only count for main task
        global_step = 0 # count for all tasks
        epoch = 0
        time_begin = time.time()
        backbone_buffer = []

        feed_batch_process_fn = create_feed_batch_process_fn(self._net_inputs)
        distribute_feeder = data_feeder(self._joint_iterator_fn, feed_batch_process_fn)
        
        while not train_finish():
            feed, mask, id = next(distribute_feeder)
            for i in range(self.dev_count):
                feed[i].update({'branch':np.array([id],dtype='int64')})
            fetch_list.append(self._switched_loss)
            rt_outputs = self.exe.run(train_program, feed=feed, fetch_list=fetch_list)
            rt_loss = rt_outputs.pop()

            rt_outputs = {k:v for k,v in zip(fetch_names, rt_outputs)}
            cur_task = instances[id]

            # backbone_rt_outputs = {k:v for k,v in rt_outputs.items() if '/' not in k}
            # backbone_buffer.append(backbone.postprocess(backbone_rt_outputs))
            
            # task_rt_outputs = {k[len(cur_task.name+'/'):]: v for k,v in rt_outputs.items() if k.startswith(cur_task.name+'/')}
            # instances[rt_task_id].task_layer['train'].postprocess(task_rt_outputs)

            global_step += 1
            cur_task.cur_train_step += 1

            cur_task_global_step = cur_task.cur_train_step + cur_task.cur_train_epoch * cur_task.steps_pur_epoch
            if cur_task.is_target and cur_task.save_infermodel_every_n_steps > 0 and cur_task_global_step % cur_task.save_infermodel_every_n_steps == 0:
                cur_task.save(suffix='.step'+str(cur_task_global_step), prog=self._pred_prog)

            if global_step % main_conf.get('print_every_n_steps', 5) == 0:
                loss = rt_loss
                loss = np.mean(np.squeeze(loss)).tolist()

                time_end = time.time()
                time_cost = time_end - time_begin

                print("Global step: {}. Task: {}, step {}/{} (epoch {}), loss: {:.3f}, speed: {:.2f} steps/s".format(
                       global_step, cur_task.name, cur_task.cur_train_step, cur_task.steps_pur_epoch, cur_task.cur_train_epoch,
                       loss, main_conf.get('print_every_n_steps', 5) / time_cost))
                time_begin = time.time()

            if cur_task.train_finish and cur_task.cur_train_step + cur_task.cur_train_epoch * cur_task.steps_pur_epoch == cur_task.expected_train_steps:
                print(cur_task.name+': train finished!')
                cur_task.save(prog=self._pred_prog)

            if 'save_ckpt_every_n_steps' in main_conf and global_step % main_conf['save_ckpt_every_n_steps'] == 0:
                save_path = os.path.join(main_conf['save_path'], 'ckpt', 
                                         "step_" + str(global_step))
                fluid.io.save_persistables(self.exe, save_path, saver_program)
                print('checkpoint has been saved at '+save_path)

        save_path = os.path.join(main_conf['save_path'], 'ckpt',
                                 "step_" + str(global_step))
        fluid.io.save_persistables(self.exe, save_path, saver_program)
        print('checkpoint has been saved at '+save_path)

        print("ALL tasks train finished, exiting...")
            
    def pred(self, task_instance, inference_model_dir=None):
        if self._for_train:
            raise Exception('This controller is a trainer. Please build a new controller with for_train=False for predicting.')

        assert isinstance(task_instance, str)
        if isinstance(inference_model_dir, str):
            assert os.path.exists(inference_model_dir), inference_model_dir+" not found."
        # if not self.has_init_pred and inference_model_dir is None:
        #     raise ValueError('infer_model_path is required for prediction.')
        if inference_model_dir is None:
            assert 'save_path' in self.mtl_conf, "one of the `inference_model_dir` and 'save_path' should be set to load inference model."
            inference_model_dir = os.path.join(self.mtl_conf['save_path'], task_instance, 'infer_model')

        instance = None
        for inst in self.instances:
            if inst.name == task_instance:
                instance = inst
                break

        if instance is None:
            raise ValueError(task_instance + ' is not a valid task_instance.')

        pred_prog = self._init_pred(instance, inference_model_dir)
                
        inst = instance
        print(inst.name+": loading data...")
        inst.reader['pred'].load_data()
        fetch_names, fetch_vars = inst.pred_fetch_list

        print('predicting...')
        feed_batch_process_fn = create_feed_batch_process_fn(inst.pred_input)
        distribute_feeder = data_feeder(inst.reader['pred'].iterator, feed_batch_process_fn, prefetch_steps=1, phase='pred')

        buf = []
        for feed, mask, id in distribute_feeder:
        
            rt_outputs = self.exe.run(pred_prog, feed, fetch_vars)
 
            nums_fake = decode_fake(len(rt_outputs[0]), mask, self.batch_size)
            while nums_fake:
                for item in rt_outputs:
                    item.pop()
                nums_fake = nums_fake - 1
    
            rt_outputs = {k:v for k,v in zip(fetch_names, rt_outputs)}
            inst.postprocess(rt_outputs, phase='pred')

        if inst.task_layer['pred'].epoch_inputs_attrs:
            reader_outputs = inst.reader['pred'].get_epoch_outputs()
        else:
            reader_outputs = None
    
        inst.epoch_postprocess({'reader':reader_outputs}, phase='pred')


if __name__ == '__main__':
    assert len(sys.argv) == 2, "Usage: python mtl_controller.py <mtl_conf_path>"
    conf_path = sys.argv[1]
    del sys.argv[1]
    controller = Controller(conf_path)
    if controller.main_conf['do_train']:
        controller.train()



__all__ = ["Controller"]
