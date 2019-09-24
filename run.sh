#!/bin/bash

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1

export CUDA_VISIBLE_DEVICES=0

if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    export CPU_NUM=1
    use_cuda=false
else
    use_cuda=true
fi

python -u mtl_run.py

