tasks="gsm8k_cot mbpp minerva_math"
nshots="8 3 4"
lengths="256 256 256"
temperatures="0 0 0"
limits="10000 10000 10000"
block_sizes="32 48 64"
block_add_thresholds="0.1 0.1 0.1"
decoded_token_thresholds="0.95 0.95 0.95"
skip_thresholds="0.9 0.9 0.9"
top_ps="none none none"
dtypes="bfloat16 bfloat16 bfloat16"
sampling_strategies="default default default"

humaneval_nshots="0"
humaneval_lengths="256"
humaneval_temperatures="0"
humaneval_limits="10000"
humaneval_diffusion_steps="256"
humaneval_block_sizes="32"
humaneval_block_add_thresholds="0.9"
humaneval_decoded_token_thresholds="0.95"
humaneval_skip_thresholds="0.95"
humaneval_top_ps="none"
humaneval_dtypes="bfloat16"
humaneval_sampling_strategies="default"

base_model=/home/xiongjing/resource_dir/models/Dream-v0-Base-7B

# Distributed training configuration
master_addr="localhost"
master_port=29522

lora_models=(
    "/home/xiongjing/resource_dir/models/D2F_Dream_Base_7B_Lora"
)

read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra LIMITS_ARRAY <<< "$limits"
read -ra BLOCK_SIZES_ARRAY <<< "$block_sizes"
read -ra BLOCK_ADD_THRESHOLDS_ARRAY <<< "$block_add_thresholds"
read -ra DECODED_TOKEN_THRESHOLDS_ARRAY <<< "$decoded_token_thresholds"
read -ra SKIP_THRESHOLDS_ARRAY <<< "$skip_thresholds"
read -ra TOP_PS_ARRAY <<< "$top_ps"
read -ra DTYPES_ARRAY <<< "$dtypes"
read -ra SAMPLING_STRATEGIES_ARRAY <<< "$sampling_strategies"

read -ra HUMANEVAL_NSHOTS_ARRAY <<< "$humaneval_nshots"
read -ra HUMANEVAL_LENGTHS_ARRAY <<< "$humaneval_lengths"
read -ra HUMANEVAL_TEMP_ARRAY <<< "$humaneval_temperatures"
read -ra HUMANEVAL_LIMITS_ARRAY <<< "$humaneval_limits"
read -ra HUMANEVAL_DIFFUSION_STEPS_ARRAY <<< "$humaneval_diffusion_steps"
read -ra HUMANEVAL_BLOCK_SIZES_ARRAY <<< "$humaneval_block_sizes"
read -ra HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY <<< "$humaneval_block_add_thresholds"
read -ra HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY <<< "$humaneval_decoded_token_thresholds"
read -ra HUMANEVAL_SKIP_THRESHOLDS_ARRAY <<< "$humaneval_skip_thresholds"
read -ra HUMANEVAL_TOP_PS_ARRAY <<< "$humaneval_top_ps"
read -ra HUMANEVAL_DTYPES_ARRAY <<< "$humaneval_dtypes"
read -ra HUMANEVAL_SAMPLING_STRATEGIES_ARRAY <<< "$humaneval_sampling_strategies"

array_length=${#TASKS_ARRAY[@]}
if [[ ${#NSHOTS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#LENGTH_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TEMP_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#LIMITS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#BLOCK_SIZES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#BLOCK_ADD_THRESHOLDS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#DECODED_TOKEN_THRESHOLDS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#SKIP_THRESHOLDS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TOP_PS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#SAMPLING_STRATEGIES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#DTYPES_ARRAY[@]} -ne $array_length ]]; then
    echo "Error: All configuration arrays must have the same length!"
    exit 1
fi

humaneval_array_length=${#HUMANEVAL_NSHOTS_ARRAY[@]}
if [[ ${#HUMANEVAL_LENGTHS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TEMP_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_LIMITS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DIFFUSION_STEPS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_BLOCK_SIZES_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_SKIP_THRESHOLDS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TOP_PS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DTYPES_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[@]} -ne $humaneval_array_length ]]; then
    echo "Error: All HumanEval configuration arrays must have the same length!"
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1
# export NCCL_P2P_DISABLE=1
for lora_model in "${lora_models[@]}"; do
    lora_model_name="$lora_model"
    echo "===================================================================="
    echo "Evaluating LoRA model: $lora_model_name"
    echo "===================================================================="

    for i in "${!HUMANEVAL_NSHOTS_ARRAY[@]}"; do
        output_path="evals_dream${lora_model_name}/humaneval-ns${HUMANEVAL_NSHOTS_ARRAY[$i]}-len${HUMANEVAL_LENGTHS_ARRAY[$i]}-temp${HUMANEVAL_TEMP_ARRAY[$i]}-limit${HUMANEVAL_LIMITS_ARRAY[$i]}-diffsteps${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]}-block${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]}-thresh${HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[$i]}-decodethresh${HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[$i]}-skip${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]}-topp${HUMANEVAL_TOP_PS_ARRAY[$i]}-dtype${HUMANEVAL_DTYPES_ARRAY[$i]}-sampling${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]}"
        echo "Running HumanEval evaluation $((i+1))/${humaneval_array_length} for $lora_model_name..."
        if [[ "${HUMANEVAL_TOP_PS_ARRAY[$i]}" == "none" ]]; then
            humaneval_model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${HUMANEVAL_LENGTHS_ARRAY[$i]},diffusion_steps=${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]},temperature=${HUMANEVAL_TEMP_ARRAY[$i]},add_bos_token=true,escape_until=true,block_size=${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]},block_add_threshold=${HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[$i]},skip_threshold=${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]},decoded_token_threshold=${HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[$i]},dtype=${HUMANEVAL_DTYPES_ARRAY[$i]},sampling_strategy=${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]},master_addr=${master_addr},master_port=${master_port},save_dir=${output_path}"
        else
            humaneval_model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${HUMANEVAL_LENGTHS_ARRAY[$i]},diffusion_steps=${HUMANEVAL_DIFFUSION_STEPS_ARRAY[$i]},temperature=${HUMANEVAL_TEMP_ARRAY[$i]},top_p=${HUMANEVAL_TOP_PS_ARRAY[$i]},add_bos_token=true,escape_until=true,block_size=${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]},block_add_threshold=${HUMANEVAL_BLOCK_ADD_THRESHOLDS_ARRAY[$i]},skip_threshold=${HUMANEVAL_SKIP_THRESHOLDS_ARRAY[$i]},decoded_token_threshold=${HUMANEVAL_DECODED_TOKEN_THRESHOLDS_ARRAY[$i]},dtype=${HUMANEVAL_DTYPES_ARRAY[$i]},sampling_strategy=${HUMANEVAL_SAMPLING_STRATEGIES_ARRAY[$i]},master_addr=${master_addr},master_port=${master_port},save_dir=${output_path}"
        fi
        CUDA_VISIBLE_DEVICES=3,4 accelerate launch --main_process_port ${master_port} --num_processes 2 eval_dream_d2f_vllm.py --model dream_lora \
            --model_args $humaneval_model_args \
            --tasks humaneval \
            --num_fewshot ${HUMANEVAL_NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path $output_path \
            --log_samples \
            --confirm_run_unsafe_code
    done

    # for i in "${!TASKS_ARRAY[@]}"; do
    #     output_path="evals_dream${lora_model_name}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-len${LENGTH_ARRAY[$i]}-temp${TEMP_ARRAY[$i]}-limit${LIMITS_ARRAY[$i]}-diffsteps${LENGTH_ARRAY[$i]}-block${BLOCK_SIZES_ARRAY[$i]}-thresh${BLOCK_ADD_THRESHOLDS_ARRAY[$i]}-decodethresh${DECODED_TOKEN_THRESHOLDS_ARRAY[$i]}-skip${SKIP_THRESHOLDS_ARRAY[$i]}-topp${TOP_PS_ARRAY[$i]}-dtype${DTYPES_ARRAY[$i]}-sampling${SAMPLING_STRATEGIES_ARRAY[$i]}"
    #     if [[ "${TOP_PS_ARRAY[$i]}" == "none" ]]; then
    #         model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},block_size=${BLOCK_SIZES_ARRAY[$i]},block_add_threshold=${BLOCK_ADD_THRESHOLDS_ARRAY[$i]},skip_threshold=${SKIP_THRESHOLDS_ARRAY[$i]},decoded_token_threshold=${DECODED_TOKEN_THRESHOLDS_ARRAY[$i]},dtype=${DTYPES_ARRAY[$i]},sampling_strategy=${SAMPLING_STRATEGIES_ARRAY[$i]},master_addr=${master_addr},master_port=${master_port},save_dir=${output_path}"
    #     else
    #         model_args="pretrained=${base_model},lora_path=${lora_model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${LENGTH_ARRAY[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=${TOP_PS_ARRAY[$i]},block_size=${BLOCK_SIZES_ARRAY[$i]},block_add_threshold=${BLOCK_ADD_THRESHOLDS_ARRAY[$i]},skip_threshold=${SKIP_THRESHOLDS_ARRAY[$i]},decoded_token_threshold=${DECODED_TOKEN_THRESHOLDS_ARRAY[$i]},dtype=${DTYPES_ARRAY[$i]},sampling_strategy=${SAMPLING_STRATEGIES_ARRAY[$i]},master_addr=${master_addr},master_port=${master_port},save_dir=${output_path}"
    #     fi
    #     CUDA_VISIBLE_DEVICES=3,4 accelerate launch --main_process_port ${master_port} --num_processes 2 eval_dream_d2f_vllm.py --model dream_lora \
    #         --model_args $model_args \
    #         --tasks ${TASKS_ARRAY[$i]} \
    #         --limit ${LIMITS_ARRAY[$i]} \
    #         --num_fewshot ${NSHOTS_ARRAY[$i]} \
    #         --batch_size 1 \
    #         --output_path $output_path \
    #         --log_samples \
    #         --confirm_run_unsafe_code
    # done
done

echo "All evaluations completed!"