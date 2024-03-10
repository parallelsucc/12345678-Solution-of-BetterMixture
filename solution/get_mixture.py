import json
import random
from pathlib import Path

num_proc=4

sample_num=65000
sample_flag=True


sample_config={
    # "HC3_Chinese_ChatGPT":10000,
    "HC3_ChatGPT":13365,#26878
    
    # "dolly":200,#233
    "instruct":1754,#3366
}


# 设置随机种子，保证结果可复现
seed = 42
random.seed(seed)

# 如果使用 NumPy
import numpy as np
np.random.seed(seed)

# 如果使用 PyTorch
import torch
torch.manual_seed(seed)

# 开发套件根目录
base_dir = Path(__file__).resolve().parent.parent
# base_dir=Path("/root/autodl-tmp/dj_mixture_challenge")

# 输入输出路径
input_dir = base_dir / "input"
ratio_path = base_dir / "output" / "sft_data" / "ratio.json"
mixture_path = base_dir / "output" / "sft_data" / "mixture.jsonl"

from data_juicer.config import init_configs
from loguru import logger
from data_juicer.core import Analyser, Executor
from datasets import concatenate_datasets
from data_juicer.core.exporter import Exporter
from transformers import AutoTokenizer
from shutil import rmtree

def process_math(input_dir):
    # data_list=["instruct","gpt4all","belle_data0.5M_cn"]
    data_list=["instruct","gpt4all"]
    file_list=[]
    for stem in data_list:
        file_list.append(input_dir / (stem+".jsonl"))

    data_pool_list=[]
    for file in file_list:
        # 清除缓存
        rmtree("/root/.cache",ignore_errors=True)

        logger.info(f'=========Processing {file}=========')
        config_file = base_dir / "solution" / "math_configs" / (str(file.stem) + ".yaml")
        dataset_path=file
        export_path = base_dir / "filtered_mathdata" / (str(file.stem)+".jsonl")
        cfg_cmd = f'--config {config_file} --dataset_path {dataset_path} --export_path {export_path}'
        args_in_cmd = cfg_cmd.split()
        cfg = init_configs(args=args_in_cmd)
        cfg.np = num_proc
        # logger.info('=========Stage 1: analyze original data=========')
        # analyzer = Analyser(cfg)
        # analyzed_dataset = analyzer.run(skip_export=True)
        logger.info('=========Stage 2: process original data=========')
        executor = Executor(cfg)
        processed_dataset = executor.run()

        shuffled_dataset=processed_dataset.shuffle(seed=seed)
        data_pool_list.append(shuffled_dataset)
        
    for i in range(len(data_pool_list)):
        data_pool_list[i]=data_pool_list[i].remove_columns(["__dj__stats__"])
        logger.warning(f'MATH: {file_list[i].name}: {len(data_pool_list[i])}')
    return data_pool_list,file_list

def process(input_dir):

    file_list=[]
    if(len(list(sample_config.keys()))!=0):
        for stem in list(sample_config.keys()):
            file_list.append(input_dir / (stem+".jsonl"))
    else:
        file_list = list(input_dir.glob("*.jsonl"))

    data_pool_list=[]
    for file in file_list:
        # 清除缓存
        rmtree("/root/.cache",ignore_errors=True)

        logger.info(f'=========Processing {file}=========')
        config_file = base_dir / "solution" / "configs" / (str(file.stem) + ".yaml")
        dataset_path=file
        export_path = base_dir / "filtered_data" / (str(file.stem)+".jsonl")
        cfg_cmd = f'--config {config_file} --dataset_path {dataset_path} --export_path {export_path}'
        args_in_cmd = cfg_cmd.split()
        cfg = init_configs(args=args_in_cmd)
        cfg.np = num_proc
        # logger.info('=========Stage 1: analyze original data=========')
        # analyzer = Analyser(cfg)
        # analyzed_dataset = analyzer.run(skip_export=True)
        logger.info('=========Stage 2: process original data=========')
        executor = Executor(cfg)
        processed_dataset = executor.run()

        shuffled_dataset=processed_dataset.shuffle(seed=seed)
        data_pool_list.append(shuffled_dataset)
        
    for i in range(len(data_pool_list)):
        data_pool_list[i]=data_pool_list[i].remove_columns(["__dj__stats__"])
        logger.warning(f'{file_list[i].name}: {len(data_pool_list[i])}')
    return data_pool_list,file_list

def sample(data_pool_list,sample_num,file_list):
    logger.info(f'=========sampling=========')

    prob_list=[]
    num_list=[]

    if(len(sample_config.keys())!=0):
        num_list=list(sample_config.values())
        sample_num=np.sum(num_list)
        prob_list=[num/sample_num for num in num_list]
    else:
        len_list=[len(data_pool) for data_pool in data_pool_list]
        len_sum=np.sum(len_list)
        
        prob_list=[len/len_sum for len in len_list]
        num_list=[int(sample_num*prob) for prob in prob_list]

    # 保存采样概率到 ratio.json
    ratio = {f.name: prob for f, prob in zip(file_list, prob_list)}
    with open(ratio_path, "w") as ratio_file:
        json.dump(ratio, ratio_file, indent=4)

    sampled_dataset_list=[]
    for i in range(len(data_pool_list)):
        sampled_dataset = data_pool_list[i].select(range(num_list[i]))
        logger.warning(f'{file_list[i].name}: {len(data_pool_list[i])} ==> {num_list[i]}')
        sampled_dataset_list.append(sampled_dataset)
    return sampled_dataset_list

def count_token(dataset,tokenizer):
    count=0
    for sample in dataset:
        count=count+len(tokenizer.tokenize(sample['text']))
    return count

def mix(sampled_dataset_list):
    logger.info(f'=========mixing=========')
    concated_datasets = concatenate_datasets(sampled_dataset_list)
    mixed_dataset=concated_datasets.shuffle(seed=seed)
    return mixed_dataset



math_data_pool_list,_=process_math(input_dir)

data_pool_list,file_list=process(input_dir)

sampled_dataset_list=[]
if(sample_flag):
    sampled_dataset_list=sample(data_pool_list,sample_num,file_list)
else:
    sampled_dataset_list=data_pool_list

# logger.info(f'=========counting=========')
# tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base", use_fast=True, trust_remote_code=True)
# for i in range(len(sampled_dataset_list)):
#     logger.warning(f'{file_list[i].name}: {count_token(sampled_dataset_list[i],tokenizer)} tokens')

sampled_dataset_list.extend(math_data_pool_list)

mixed_dataset=mix(sampled_dataset_list)

logger.info(f'=========exporting=========')
exporter = Exporter(export_path=str(mixture_path),
                    export_shard_size=0,#to a single file
                    num_proc=num_proc,
                    export_stats=True)

exporter.export(mixed_dataset)

