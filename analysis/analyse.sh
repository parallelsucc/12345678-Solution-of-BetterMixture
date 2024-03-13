#!/bin/bash

# 设置配置文件目录
config_dir="./config"

# 遍历配置文件目录下的所有yaml文件
for yaml_file in "$config_dir"/*.yaml; do
    rm -r ~/.cache/huggingface
    rm -r ~/.cache/gdown
    rm -r ~/.cache/matplotlib
    if [ -f "$yaml_file" ]; then
        echo "Running with $yaml_file"
        # 调用并传递yaml文件作为参数
        dj-analyze --config "$yaml_file"
    fi
done