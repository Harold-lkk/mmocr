source /usr/local/env/pat_latest

#env

export LD_LIBRARY_PATH=/usr/local/gcc-5.4/lib64:/usr/local/binutils-2.27/lib:/usr/local/nccl-2.4.7-cuda9.0/lib/:/usr/local/openmpi-2.1.6-cuda9.0/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/env/miniconda3.6/lib:/usr/local/openmpi-2.1.6-cuda9.0/lib:/usr/local/openmpi-2.1.6-cuda9.0/lib:/usr/local/binutils-2.27/lib:/usr/local/openmpi-2.1.6-cuda9.0/lib:/usr/local/nccl-2.4.7-cuda9.0/lib:/usr/local/pplbase/lib:/usr/local/PatATen/lib:/usr/local/cuda/lib64:/usr/local/libmemcached/lib/:/usr/local/memcached_client/lib/:/usr/local/boost/lib/:/usr/lib64/:/usr/local/binutils-2.27/lib:/usr/local/openmpi-2.1.6-cuda9.0/lib:/usr/local/nccl-2.4.7-cuda9.0/lib:/usr/local/pplbase/lib:/usr/local/PatATen/lib:/usr/local/cuda/lib64:/usr/local/libmemcached/lib/:/usr/local/memcached_client/lib/:/usr/local/boost/lib/:/usr/lib64/:/usr/local/openmpi-2.1.6-cuda9.0/lib:/usr/local/nccl-2.4.7-cuda9.0/lib:/usr/local/pplbase/lib:/usr/local/PatATen/lib:/usr/local/cuda/lib64:/usr/local/libmemcached/lib/:/usr/local/memcached_client/lib/:/usr/local/boost/lib/:/usr/lib64/:/usr/lib64/:/usr/local/openmpi-2.1.6-cuda9.0/lib:/usr/local/nccl-2.4.7-cuda9.0/lib:/usr/local/pplbase/lib:/usr/local/PatATen/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export RANK=${OMPI_COMM_WORLD_RANK}
export PAVI_GW_URL=${input_parameter_PAVI_GW_URL}
export PAVI_DEBUG=false
export PAVI_HTTP_VERIFY=false
pip install -v -e .
pip install -r requirements.txt


PYTHONPATH='.':${PYTHONPATH} PARROTS_EXEC_MODE=SYNC PARROTS_BT_DEPTH=15 PARROTS_ALIGN_TORCH=1 sh tools/dist_train.sh configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_0e_textocr.py checkpoints/ 8