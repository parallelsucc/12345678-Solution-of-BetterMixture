#!/bin/bash

# Enable offline mode to speed up loading
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Get the directory of the current training directory
WORK_DIR=$(echo `cd $(dirname $0); pwd | xargs dirname`)

# Function for printing log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function for displaying help information
usage() {
	echo
    echo "Usage: $0 --model MODEL_PATH --data DATA_PATH --output OUTPUT_PATH"
    echo "  --model    Path to the base model."
    echo "  --data     Path to the data."
    echo "  --output   Path to the output."
    echo "  --dtype    Compute type (default auto)."
    echo "  --int8     Enable load_in_8bit (default True)."
    echo "  --pack     Enable sft_packing (default True)."
    echo "  --batch_size Batchsize per gpu (default 1)."
    echo "  --lr            Learning rate (default 1e-5)."
    echo "  --ds_stage      DeepSpeed zero stage (default 2)."
    echo "  --ds_offload    DeepSpeed offload opt and param (default False)."
    exit 1
}

# Parse command-line arguments
compute_dtype="auto"
load_in_8bit="True"
sft_packing="True"
batch_size=1
learning_rate=1e-5
template=alpaca
ds_stage=2
ds_offload="False"
lora_rank=8
lora_alpha=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --model)
            model_path="$2"
            shift 2
            ;;
        --data)
            data_path="$2"
            shift 2
            ;;
        --output)
            output_path="$2"
            shift 2
            ;;
        --dtype)
            compute_dtype="$2"
            shift 2
            ;;
        --int8)
            load_in_8bit="$2"
            shift 2
            ;;
        --pack)
            sft_packing="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --ds_stage)
            ds_stage="$2"
            shift 2
            ;;
        --ds_offload)
            ds_offload="$2"
            shift 2
            ;;
        --lr)
            learning_rate="$2"
            shift 2
            ;;
        *)
            log "Invalid option: $1"
            usage
            ;;
    esac
done

# Check required parameters
if [ -z "${model_path}" ] || [ -z "${data_path}" ] || [ -z "${output_path}" ]; then
    log "Error: Missing required parameters."
    usage
fi

# Post-process paths
mkdir -p "${output_path}" || { log "Error: Failed to create output directory."; exit 1; }

log "[MODEL] ${model_path}"
log "[DATA] ${data_path}"
log "[OUT] ${output_path}"
log "[DTYPE] ${compute_dtype}"
log "[INT8] ${load_in_8bit}"
log "[SFT_PACKING] ${sft_packing}"
log "[BATCH_SIZE] ${batch_size}"
log "[LR] ${learning_rate}"
log "[DS_STAGE] ${ds_stage}"
log "[DS_OFFLOAD] ${ds_offload}"

if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" = "all" ]; then
    num_gpus=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l | tr -d ' ')
    gpu_list=""
    for (( i=0; i<num_gpus; i++ )); do
        if [ $i -gt 0 ]; then
            gpu_list="$gpu_list,"
        fi
        gpu_list="$gpu_list$i"
    done
    export CUDA_VISIBLE_DEVICES=$gpu_list
else
    num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

grad_acc_steps=`expr 256 / ${batch_size} / ${num_gpus}`

dtype=""
if [ "$compute_dtype" = "bfloat16" ]; then
    dtype="--bf16"
elif [ "$compute_dtype" = "float16" ]; then
    dtype="--fp16"
fi

# Select deepspeed config
if [[ "$ds_offload" == "True" ]]; then
    ds_config=$WORK_DIR/examples/ds_config_stage${ds_stage}_offload.json
else
    ds_config=$WORK_DIR/examples/ds_config_stage${ds_stage}.json
fi

# Set launch commond
master_port=`python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`
cmd="deepspeed --include=localhost:${CUDA_VISIBLE_DEVICES} --master_port ${master_port}"
unset CUDA_VISIBLE_DEVICES

# Append all the arguments to the command variable.
cmd+=" $WORK_DIR/src/train_bash.py"
cmd+=" --deepspeed ${ds_config}"
cmd+=" --stage sft"
cmd+=" --model_name_or_path ${model_path}"
cmd+=" --do_train"
cmd+=" --dataset ${data_path}"
cmd+=" --finetuning_type lora"
cmd+=" --lora_alpha ${lora_alpha}"
cmd+=" --lora_rank ${lora_rank}"
cmd+=" --template ${template}"
cmd+=" --lora_target W_pack"
cmd+=" --output_dir ${output_path}"
cmd+=" --overwrite_cache"
cmd+=" --per_device_train_batch_size ${batch_size}"
cmd+=" --gradient_accumulation_steps ${grad_acc_steps}"
cmd+=" --lr_scheduler_type cosine"
cmd+=" --logging_steps 1"
cmd+=" --save_steps 100000"
cmd+=" --overwrite_output_dir"
cmd+=" --learning_rate ${learning_rate}"
cmd+=" --num_train_epochs 3.0"
cmd+=" --plot_loss"
cmd+=" --weight_decay 0"
cmd+=" --warmup_ratio 0.03"
cmd+=" --max_tokens 1e7"
cmd+=" --load_in_8bit ${load_in_8bit}"
cmd+=" --sft_packing ${sft_packing}"
cmd+=" ${dtype}"

# Execute the command and pipe the output to tee for logging.
$cmd | tee "${output_path}/training_log.txt"