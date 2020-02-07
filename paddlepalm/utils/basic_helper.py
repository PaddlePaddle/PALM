# coding=utf-8
import os
import json
import yaml
from .config_helper import PDConfig
import logging
from paddle import fluid

def get_basename(f):
    return os.path.splitext(f)[0]


def get_suffix(f):
    return os.path.splitext(f)[-1]


def parse_yaml(f, asdict=True, support_cmd_line=False):
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


def parse_json(f, asdict=True, support_cmd_line=False):
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
            

def parse_list(string, astype=str):
    assert isinstance(string, str), "{} is not a string.".format(string)
    if ',' not in string:
        return [astype(string)]
    string = string.replace(',', ' ')
    return [astype(i) for i in string.split()]


def try_float(s):
    try:
        float(s)
        return(float(s))
    except:
        return s


# TODO: 增加None机制，允许hidden size、batch size和seqlen设置为None
def check_io(in_attr, out_attr, strict=False, in_name="left", out_name="right"):
    for name, attr in in_attr.items():
        assert name in out_attr, in_name+': '+name+' not found in '+out_name
        if attr != out_attr[name]:
            if strict:
                raise ValueError(name+': shape or dtype not consistent!')
            else:
                logging.warning('{}: shape or dtype not consistent!\n{}:\n{}\n{}:\n{}'.format(name, in_name, attr, out_name, out_attr[name]))


def encode_inputs(inputs, scope_name, sep='.', cand_set=None):
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


def decode_inputs(inputs, scope_name, sep='.', keep_unk_keys=True):
    outputs = {}
    for name, value in inputs.items():
        # var for backbone are also available to tasks
        if keep_unk_keys and sep not in name:
            outputs[name] = value
        # var for this inst
        if name.startswith(scope_name+'.'):
            outputs[name[len(scope_name+'.'):]] = value
    return outputs


def build_executor(on_gpu):
    if on_gpu:
        place = fluid.CUDAPlace(0)
        # dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        # dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    # return fluid.Executor(place), dev_count
    return fluid.Executor(place)


def fit_attr(conf, fit_attr, strict=False):
    for i, attr in fit_attr.items():
        if i not in conf:
            if strict:
                raise Exception('Argument {} is required to create a controller.'.format(i))
            else:
                continue
        conf[i] = attr(conf[i])
    return conf
