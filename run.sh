#!/bin/bash

eval "$(conda shell.bash hook)"

# env
conda activate dd

# 检查checkpoint文件夹是否存在
if [ ! -d "./checkpoint" ]; then
    # 如果不存在，则创建它
    mkdir ./checkpoint
    echo "Create 'checkpoint' folder."
else
    echo "'checkpoint' folder already exists."
fi

# Train
python diffusion_distillation.py

latest_checkpoint = ""

# 检查 /tmp/flax_ckpt/checkpoint/ 文件夹是否存在
if [ ! -d "/tmp/flax_ckpt/checkpoint/" ]; then
    echo "/tmp/flax_ckpt/checkpoint/ 文件夹不存在"
else
    # 查找以 "checkpoint_" 开头的所有文件
    checkpoint_files=($(ls /tmp/flax_ckpt/checkpoint/checkpoint_* 2>/dev/null))

    if [ ${#checkpoint_files[@]} -eq 0 ]; then
        echo "没有找到以 'checkpoint_' 开头的文件"
    else
        # 找到最近修改的 "checkpoint_" 文件
        latest_checkpoint=$(ls -t /tmp/flax_ckpt/checkpoint/checkpoint_* | head -n 1)
        echo "最近修改的 'checkpoint_' 文件路径: $latest_checkpoint"
    fi
fi

# 创建临时文件
tmp_output=$(mktemp)

# 运行 Python 脚本并将输出重定向到终端和临时文件
python sample_origin.py --num_imgs 1024 --batchsize 64 --startbatch 0 --db_path data/mnist_origin_debug --ckpt_path $latest_checkpoint | tee "$tmp_output"

# 从临时文件中提取 db_path
db_path=$(grep "Write" "$tmp_output" | awk -F ' ' '{print $5}')

echo "db_path: $db_path"

# 删除临时文件
rm "$tmp_output"
