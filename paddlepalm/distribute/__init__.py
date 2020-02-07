from paddle import fluid
import os
import multiprocessing

gpu_dev_count = int(fluid.core.get_cuda_device_count())
cpu_dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

from .reader import yield_pieces, data_feeder, decode_fake

