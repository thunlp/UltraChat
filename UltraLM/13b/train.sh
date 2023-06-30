GPUS_PER_NODE=4
WORLD_SIZE=1

OPTS=""
OPTS+=" --logging_step 50" 
OPTS+=" --batch_size_per_device 2"
OPTS+=" --save_step 20"
OPTS+=" --epochs 5"
OPTS+=" --train-iters 1000000"
OPTS+=" --start-step 0"
OPTS+=" --model_name_or_path huggyllama/llama-13b"
OPTS+=" --model ultralm-13b"
OPTS+=" --tensorboard ./tensorboard/ultralm-13b/"`date +"%Y%m%d%H%M%S"`
OPTS+=" --save_dir ./checkpoints/"


CMD="torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} train.py ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"
$CMD
# --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}  --rdzv_id=1 --rdzv_backend=c10d 