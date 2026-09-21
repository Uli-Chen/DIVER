#!/bin/bash

# 获取显存占用最低的GPU
select_lowest_mem_gpu() {
    # 获取GPU信息并格式化为：gpu_id,mem_used,mem_total,gpu_util
    local gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                      --format=csv,noheader,nounits 2>/dev/null)

    # 检查是否获取到GPU信息
    if [ -z "$gpu_info" ]; then
        echo "错误：无法获取GPU信息"
        exit 1
    fi

    # 过滤出 GPU 2-7（包含边界）
    local filtered_gpus=$(echo "$gpu_info" | awk -F',' '$1 >= 2 && $1 <= 7')

    # 检查过滤后是否有可用GPU
    if [ -z "$filtered_gpus" ]; then
        echo "错误：GPU 2-7 中未找到可用设备"
        exit 1
    fi

    # 按显存使用量升序排序（第2列数值排序）
    local sorted_gpus=$(echo "$filtered_gpus" | sort -t, -n -k2)

    # 选择显存最低的GPU
    local best_gpu=$(echo "$sorted_gpus" | head -n1 | cut -d, -f1)
    
    # 验证选择结果
    if [ -z "$best_gpu" ]; then
        echo "错误：未能选择有效GPU"
        exit 1
    fi

    echo "$best_gpu"
}

# 参数选项列表
datasets=("poke" "med" "harry")
# defense_policies=("input_filter" "output_filter" "intention_detection" "defensive_prompt" "no_defense")
defense_policies=("no_defense")
llm_choices=("gpt" "deepseek" "llama")
# llm_choices=("deepseek" "llama")
emb_models=("bge" "mpnet")

# 实验计数器
total=$(( ${#datasets[@]} * ${#defense_policies[@]} * ${#llm_choices[@]} * ${#emb_models[@]} ))
current=0

# 创建日志目录
log_dir="/home/guest/data/IKEA/auto_dgea_experiment_logs"
mkdir -p $log_dir

# 主循环
for llm_choice in "${llm_choices[@]}"; do
    for emb_model in "${emb_models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for defense_policy in "${defense_policies[@]}"; do
                ((current++))
                
                # 自动选择显存最低的GPU
                gpu_id=$(select_lowest_mem_gpu)
                gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i $gpu_id)
                mem_used=$(echo "$gpu_info" | cut -d, -f2)
                mem_total=$(echo "$gpu_info" | cut -d, -f3)
                
                timestamp=$(date "+%Y%m%d-%H%M%S")
                log_file="${log_dir}/${timestamp}_${llm_choice}_${emb_model}_${dataset}_${defense_policy}.log"

                export CUDA_VISIBLE_DEVICES=$gpu_id
                # 构造命令
                cmd=(
                    /home/guest/conda3/envs/KG_extract/bin/python /home/guest/rag-framework/dgea_pipeline.py
                    --dataset "$dataset"
                    --defense_policy "$defense_policy"
                    --llm_choice "$llm_choice"
                    --emb_model_choice "$emb_model"
                )

                # 打印进度
                echo -e "\n[进度] $current/$total | GPU-$gpu_id (${mem_used}/${mem_total}MB) | $(date)"
                echo "[组合] dataset=$dataset, defense=$defense_policy"
                echo "       llm=$llm_choice, emb=$emb_model"
                echo "[命令] ${cmd[@]}"

                # 执行命令并记录日志
                eval "${cmd[@]}" > "$log_file" 2>&1

                # 添加冷却时间
                sleep 10
            done
        done
    done
done

echo -e "\n所有实验执行完成！日志保存在: $log_dir"