export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_fraction_of_gpu_memory_to_use=0.1
export FLAGS_eager_delete_tensor_gb=0

python demo1.py

