# encoding=utf8

import os
import sys
import time
import argparse
import importlib
import collections
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid

from utils.configure import PDConfig
from utils.placeholder import Placeholder 
from utils.configure import JsonConfig, ArgumentGroup, print_arguments
from utils.init import init_pretraining_params, init_checkpoint

sys.path.append("reader")
import joint_reader
from joint_reader import create_reader

sys.path.append("optimizer")
sys.path.append("paradigm")
sys.path.append("backbone")

TASKSET_PATH="config"

def train(multitask_config): 

    # load task config
    print("Loading multi_task configure...................")
    args = PDConfig(yaml_file=[multitask_config])
    args.build()

    index = 0
    reader_map_task = dict()
    task_args_list = list()
    reader_args_list = list()
    id_map_task = {index: args.main_task}
    print("Loading main task configure....................")
    main_task_name = args.main_task
    task_config_files = [i for i in os.listdir(TASKSET_PATH) if i.endswith('.yaml')]
    main_config_list = [config for config in task_config_files if config.split('.')[0] == main_task_name]
    main_args = None
    for config in main_config_list: 
        main_yaml = os.path.join(TASKSET_PATH, config)
        main_args = PDConfig(yaml_file=[multitask_config, main_yaml])
        main_args.build()
        main_args.Print()
        if not task_args_list or main_task_name != task_args_list[-1][0]: 
            task_args_list.append((main_task_name, main_args))
        reader_args_list.append((config.strip('.yaml'), main_args))
        reader_map_task[config.strip('.yaml')] = main_task_name

    print("Loading auxiliary tasks configure...................")
    aux_task_name_list = args.auxiliary_task.strip().split()
    for aux_task_name in aux_task_name_list: 
        index += 1
        id_map_task[index] = aux_task_name
        print("Loading %s auxiliary tasks configure......." % aux_task_name)
        aux_config_list = [config for config in task_config_files if config.split('.')[0] == aux_task_name]
        for aux_yaml in aux_config_list: 
            aux_yaml = os.path.join(TASKSET_PATH, aux_yaml)
            aux_args = PDConfig(yaml_file=[multitask_config, aux_yaml])
            aux_args.build()
            aux_args.Print()
            if aux_task_name != task_args_list[-1][0]: 
                task_args_list.append((aux_task_name, aux_args))
            reader_args_list.append((aux_yaml.strip('.yaml'), aux_args))
            reader_map_task[aux_yaml.strip('.yaml')] = aux_task_name

    # import tasks reader module and build joint_input_shape
    input_shape_list = []
    reader_module_dict = {}
    input_shape_dict = {}
    for params in task_args_list: 
        task_reader_mdl = "%s_reader" % params[0]
        reader_module = importlib.import_module(task_reader_mdl)
        reader_servlet_cls = getattr(reader_module, "get_input_shape")
        reader_input_shape = reader_servlet_cls(params[1])
        reader_module_dict[params[0]] = reader_module
        input_shape_list.append(reader_input_shape)
        input_shape_dict[params[0]] = reader_input_shape
    train_input_shape, test_input_shape, task_map_id = joint_reader.joint_input_shape(input_shape_list)

    # import backbone model
    backbone_mdl = args.backbone_model
    backbone_cls = "Model"
    backbone_module = importlib.import_module(backbone_mdl)
    backbone_servlet = getattr(backbone_module, backbone_cls)

    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)
    startup_prog = fluid.default_startup_program()

    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train: 
        #create joint pyreader
        print('creating readers...')
        gens = []
        main_generator = ""
        for params in reader_args_list: 
            generator_cls = getattr(reader_module_dict[reader_map_task[params[0]]], "DataProcessor")
            generator_inst = generator_cls(params[1])
            reader_generator = generator_inst.data_generator(phase='train', shuffle=True, dev_count=dev_count)
            if not main_generator: 
                main_generator = generator_inst
            gens.append((reader_generator, params[1].mix_ratio, reader_map_task[params[0]]))
        joint_generator, train_pyreader, model_inputs = create_reader("train_reader", train_input_shape, True, task_map_id, gens)

        train_pyreader.decorate_tensor_provider(joint_generator)

        # build task inputs 
        task_inputs_list = []
        main_test_input = []
        task_id = model_inputs[0]
        backbone_inputs = model_inputs[task_map_id[0][0]: task_map_id[0][1]]
        for i in range(1, len(task_map_id)): 
            task_inputs = backbone_inputs + model_inputs[task_map_id[i][0]: task_map_id[i][1]]
            task_inputs_list.append(task_inputs)

        # build backbone model
        print('building model backbone...')
        conf = vars(args)
        if args.pretrain_config_path is not None:
            model_conf = JsonConfig(args.pretrain_config_path).asdict()
            for k, v in model_conf.items():
                if k in conf:
                    assert k == conf[k], "ERROR: argument {} in pretrain_model_config is NOT consistent with which in main.yaml"
            conf.update(model_conf)

        backbone_inst = backbone_servlet(conf, is_training=True)
       
        print('building task models...')
        num_train_examples = main_generator.get_num_examples()
        if main_args.in_tokens:
            max_train_steps = int(main_args.epoch * num_train_examples) // (
                    main_args.batch_size // main_args.max_seq_len) // dev_count
        else:
            max_train_steps = int(main_args.epoch * num_train_examples) // (
                main_args.batch_size) // dev_count
        mix_ratio_list = [task_args[1].mix_ratio for task_args in task_args_list]
        args.max_train_steps = int(max_train_steps * (sum(mix_ratio_list) / main_args.mix_ratio))
        print("Max train steps: %d" % max_train_steps)

        build_strategy = fluid.BuildStrategy()
        train_program = fluid.default_main_program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                
                backbone_inst.build_model(backbone_inputs)
                all_loss_list = []

                for i in range(len(task_args_list)): 
                    task_name = task_args_list[i][0]
                    task_args = task_args_list[i][1]

                    if hasattr(task_args, 'paradigm'):
                        task_net = task_args.paradigm
                    else:
                        task_net = task_name

                    task_net_mdl = importlib.import_module(task_net)
                    task_net_cls = getattr(task_net_mdl, "create_model")
                    output_tensor = task_net_cls(task_inputs_list[i], base_model=backbone_inst, is_training=True, args=task_args)
                    loss_cls = getattr(task_net_mdl, "compute_loss")
                    task_loss = loss_cls(output_tensor, task_args)
                    all_loss_list.append(task_loss)
                    num_seqs = output_tensor['num_seqs']

                task_one_hot = fluid.layers.one_hot(task_id, len(task_args_list))
                all_loss = fluid.layers.concat(all_loss_list, axis=0)
                loss = fluid.layers.reduce_sum(task_one_hot * all_loss)
              
                programs = [train_program, startup_prog]
                optimizer_mdl = importlib.import_module(args.optimizer)
                optimizer_inst = getattr(optimizer_mdl, "optimization")
                optimizer_inst(loss, programs, args=args)
                
                loss.persistable = True
                num_seqs.persistable = True

                ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)
                ema.update()

        train_compiled_program = fluid.CompiledProgram(train_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)

    if args.do_predict:
        conf = vars(args)
        if args.pretrain_config_path is not None:
            model_conf = JsonConfig(args.pretrain_config_path).asdict()
            for k, v in model_conf.items():
                if k in conf:
                    assert v == conf[k], "ERROR: argument {} in pretrain_model_config is NOT consistent with which in main.yaml".format(k)
            conf.update(model_conf)
        mod = reader_module_dict[main_task_name]
        DataProcessor = getattr(mod, 'DataProcessor')
        predict_processor = DataProcessor(main_args)
        test_generator = predict_processor.data_generator(
            phase='predict',
            shuffle=False,
            dev_count=dev_count)

        new_test_input_shape = input_shape_dict[main_task_name][1]['backbone'] + input_shape_dict[main_task_name][1]['task']
        assert new_test_input_shape == test_input_shape
        build_strategy = fluid.BuildStrategy()
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                placeholder = Placeholder(test_input_shape)
                test_pyreader, model_inputs = placeholder.build(
                    capacity=100, reader_name="test_reader")

                test_pyreader.decorate_tensor_provider(test_generator)

                # create model
                backbone_inst = backbone_servlet(conf, is_training=False)

                backbone_inst.build_model(model_inputs)

                task_net_mdl = importlib.import_module(main_task_name)
                task_net_cls = getattr(task_net_mdl, "create_model")
                postprocess = getattr(task_net_mdl, "postprocess")
                global_postprocess = getattr(task_net_mdl, "global_postprocess")
                output_tensor = task_net_cls(model_inputs, base_model=backbone_inst, is_training=False, args=main_args)

                if 'ema' not in dir():
                    ema = fluid.optimizer.ExponentialMovingAverage(args.ema_decay)

                pred_fetch_names = []
                fetch_vars = []
                for i,j in output_tensor.items():
                    pred_fetch_names.append(i)
                    fetch_vars.append(j)
                for var in fetch_vars:
                    var.persistable = True
                pred_fetch_list = [i.name for i in fetch_vars]


        test_prog = test_prog.clone(for_test=True)
        test_compiled_program = fluid.CompiledProgram(test_prog).with_data_parallel(
            build_strategy=build_strategy)

    exe.run(startup_prog)

    if args.do_train:
        if args.pretrain_model_path:
            init_pretraining_params(
                exe,
                args.pretrain_model_path,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
        if args.checkpoint_path:
            if os.path.exists(args.checkpoint_path):
                init_checkpoint(
                    exe,
                    args.checkpoint_path,
                    main_program=startup_prog,
                    use_fp16=args.use_fp16)
            else:
                os.makedirs(args.checkpoint_path)

    elif args.do_predict:
        if not args.checkpoint_path:
            raise ValueError("args 'checkpoint_path' should be set if"
                             "only doing prediction!")
        init_checkpoint(
            exe,
            args.checkpoint_path,
            main_program=test_prog,
            use_fp16=args.use_fp16)

    if args.do_train:
        print('start training...')
        train_pyreader.start()

        steps = 0
        total_cost, total_num_seqs = [], []
        time_begin = time.time()
        while True:
            try:
                steps += 1
                if steps % args.skip_steps == 0:
                    fetch_list = [loss.name, num_seqs.name, task_id.name]
                else:
                    fetch_list = []

                outputs = exe.run(train_compiled_program, fetch_list=fetch_list)

                if steps % args.skip_steps == 0:
                    np_loss, np_num_seqs, np_task_id = outputs
                    total_cost.extend(np_loss * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_example, epoch = main_generator.get_train_progress()
                   
                    cur_task_name = id_map_task[np_task_id[0][0]]
                    print("epoch: %d, task_name: %s, progress: %d/%d, step: %d, loss: %f, "
                          "speed: %f steps/s" %
                          (epoch, cur_task_name, current_example, num_train_examples, steps,
                           np.sum(total_cost) / np.sum(total_num_seqs),
                           args.skip_steps / used_time))
                    total_cost, total_num_seqs = [], []
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoint_path,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
                if steps == max_train_steps:
                    save_path = os.path.join(args.checkpoint_path,
                                             "step_" + str(steps) + "_final")
                    fluid.io.save_persistables(exe, save_path, train_program)
                    break
            except paddle.fluid.core.EOFException as err:
                save_path = os.path.join(args.checkpoint_path,
                                         "step_" + str(steps) + "_final")
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    if args.do_predict:
        print('start predicting...')
        cnt = 0
        if args.use_ema:
            with ema.apply(exe):
                test_pyreader.start()
                pred_buf = []
                while True:
                    try:
                        fetch_res = exe.run(fetch_list=pred_fetch_list, program=test_compiled_program)
                        cnt += 1
                        if cnt % 200 == 0:
                            print('predicting {}th batch...'.format(cnt))
                        fetch_dict = {}
                        for key,val in zip(pred_fetch_names, fetch_res):
                            fetch_dict[key] = val
                        res = postprocess(fetch_dict)
                        if res is not None:
                            pred_buf.extend(res)
                    except fluid.core.EOFException:
                        test_pyreader.reset()
                        break
                global_postprocess(pred_buf, predict_processor, args, main_args)
        else:
            test_pyreader.start()
            pred_buf = []
            while True:
                try:
                    fetch_res = exe.run(fetch_list=pred_fetch_list, program=test_compiled_program)
                    cnt += 1
                    if cnt % 200 == 0:
                        print('predicting {}th batch...'.format(cnt))
                    fetch_dict = {}
                    for key,val in zip(pred_fetch_names, fetch_res):
                        fetch_dict[key] = val
                    res = postprocess(fetch_dict)
                    if res is not None:
                        pred_buf.extend(res)
                except fluid.core.EOFException:
                    test_pyreader.reset()
                    break
            global_postprocess(pred_buf, predict_processor, args, main_args)


if __name__ == '__main__':

    multitask_config = "mtl_config.yaml"
    train(multitask_config)
