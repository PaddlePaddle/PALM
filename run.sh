#!/bin/bash

# for gpu memory optimization
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1

export CUDA_VISIBLE_DEVICES=0

if [[ ! -d pretrain_model/bert ]]; then
    bash download_pretrain.sh bert
fi

if [[ ! -d pretrain_model/ernie ]]; then
    bash download_pretrain.sh ernie
fi

python -u mtl_run.py

