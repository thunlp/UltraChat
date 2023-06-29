GPUS_PER_NODE=8
WORLD_SIZE=2

OPTS=""
OPTS+=" --logging_step 50" 
OPTS+=" --batch_size_per_device 2"
OPTS+=" --save_step 200"
OPTS+=" --epochs 5"
OPTS+=" --train-iters 1000000"
OPTS+=" --start-step 0"
OPTS+=" --model_name_or_path huggyllama/llama-13b"
OPTS+=" --model ultralm-13b"
OPTS+=" --tensorboard ./tensorboard/ultralm-13b/"`date +"%Y%m%d%H%M%S"`
OPTS+=" --save_dir /data/checkpoints/ultralm"


CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD
