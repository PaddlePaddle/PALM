export CUDA_VISIBLE_DEVICES=0

python -u demo2.py 
# GLOG_vmodule=lookup_table_op=4 python -u demo2.py > debug2.log 2>&1

