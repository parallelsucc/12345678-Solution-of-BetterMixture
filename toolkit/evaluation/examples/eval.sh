#!/bin/bash

# Enable offline mode to speed up loading
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Get the directory of the current script
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MAIN_SCRIPT="$SCRIPT_DIR/../main.py"
SUMM_SCRIPT="$SCRIPT_DIR/../summarize.py"

# Function for printing log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function for displaying help information
usage() {
	echo
    echo "Usage: $0 --mode MODE --model MODEL_PATH --data DATA_PATH --output OUTPUT_PATH"
    echo "  --mode     Mode of the evaluation."
    echo "  --model    Path to the base model."
    echo "  --lora     Path to the lora model (default '')."
    echo "  --dtype    Compute type (default auto)."
    echo "  --int8     Enable load_in_8bit (default False)."
    echo "  --data     Path to the data."
    echo "  --seqlen   Maximum sequence length. (default '')"
    echo "  --output   Path to the output."
    exit 1
}

# Parse command-line arguments
lora_path=""
compute_dtype="auto"
load_in_8bit="False"
max_length=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --model)
            model_path="$2"
            shift 2
            ;;
        --seqlen)
            max_length="$2"
            shift 2
            ;;
        --lora)
            lora_path="$2"
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
        --data)
            data_path="$2"
            shift 2
            ;;
        --output)
            output_path="$2"
            shift 2
            ;;
        *)
            log "Invalid option: $1"
            usage
            ;;
    esac
done

# Check required parameters
if [ -z "${mode}" ] || [ -z "${model_path}" ] || [ -z "${data_path}" ] || [ -z "${output_path}" ]; then
    log "Error: Missing required parameters."
    usage
fi

# Post-process paths
data_path="${data_path}/${mode}"
output_path="${output_path}/${mode}"
mkdir -p "${output_path}" || { log "Error: Failed to create output directory."; exit 1; }

log "[MODEL] ${model_path}"
log "[LORA] ${lora_path}"
log "[DTYPE] ${compute_dtype}"
log "[INT8] ${load_in_8bit}"
log "[DATA] ${data_path}"
log "[OUT] ${output_path}"

# Define tasks and fewshot numbers using an indexed array
task_fewshot=(
    arc_challenge         25
    hellaswag             25
    truthfulqa_mc         0
    hendrycksTest-*       5
    cmmlu-*               5
    gsm8k                 5
    scrolls_summscreenfd  0
)

# Start evaluation
for ((i=0; i<${#task_fewshot[@]}; i+=2)); do
    task=${task_fewshot[$i]}
    fewshot=${task_fewshot[$i+1]}
    log "[TASK] ${task}: ${fewshot}-shot"
    python "${MAIN_SCRIPT}" \
        --model=hf-causal \
        --model_args="pretrained=${model_path},peft=${lora_path},dtype=${compute_dtype},load_in_8bit=${load_in_8bit},max_length=${max_length},trust_remote_code=True" \
        --tasks="${task}" \
        --num_fewshot="${fewshot}" \
        --device=cuda:0 \
        --batch_size=1 \
        --no_cache \
        --local_data_path="${data_path}" \
        --output_path="${output_path}/${task}.json" \
        --detail_output_path="${output_path}/detail" \
        $(test "${mode}" = "board" && echo "--infer_only" || echo "")
done

# Summarize results
if [[ "${mode}" != "board" ]]; then
    log "[SUMMARIZE]"
    python "${SUMM_SCRIPT}" \
        --output_path "${output_path}"
else
    log "[Done]"
fi

