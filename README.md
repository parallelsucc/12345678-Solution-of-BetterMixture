## TODO：更新迭代实验记录及相关分析

### 1. 添加筛选数学的算子
    math_num_filter.py
* 将该文件添加到 data_juicer/ops/filter/
* 修改系列相关文件
* 重新安装 data_juicer
> 通过正则表达式 __'\d+(\s*[-+*/]\s*\d+)+\s*=\s*\d+'__ 筛选与数学相关的数据  
### 2. 添加去除网页代码的算子
    normal_filter.py
* 添加方法同上
> 通过 "<img" 等筛除网页源码数据，并去除 input 和 instruct 异常短(长度不大于3)的数据
### 3. 修改 ppl 算子，添加 min_ppl
    perplexity_filter.py
* 修改完成后，重新安装 data_juicer
> 筛除 ppl < min_ppl 的数据
### 4. 修改 entry.env
* INT8=True
* SFT_PACKING=False
### 5. 各个文件的筛选策略
    configs/HC3_ChatGPT.yaml
> 筛选完成后，剩余 __13365__ 条数据  
> ppl 筛选依据：整体数据的 25% - 75% (min_ppl: 148.9 max_ppl: 279.1)  
> 长文本有 instruct, 且 HC3_ChatGPT 长文本表现一般, 所以 HC3_ChatGPT token_num 取 10 - 500

    configs/instruct.yaml
> 筛选完成后，剩余 __1754__ 条数据  
> ppl 筛选依据：整体数据的 0 - 75% (max_ppl: 1170.6)  
> instruct 长文本表现较好，百川 7B 模型最大 token 为 4096，为避免过长，instruct toekn_num 取 1000 - 4000

    math_configs/gpt4all.yaml
> 筛选完成后，剩余 __2339__ 条数据  
> 根据添加的数学算子，筛选 gpt4all 中与数学相关的数据

    math_configs/instruct.yaml
> 筛选完成后，剩余 __3794__ 条数据  
> 根据添加的数学算子，筛选 instruct 中与数学相关的数据
### 6. get_mixture.py 执行过程
1. 根据 __math_configs/__ 的算子筛选 __gpt4all__ 和 __instruct__
2. 根据 __configs/__ 的算子筛选 __HC3_ChatGPT__ 和 __instruct__
3. 将 4 个数据集 (instruct 被筛选两次) 的筛选结果混合
4. 对混合结果 shuffle(seed=42)
> 取全部筛选后的结果，即  
> HC3_ChatGPT __13365__  
> instruct __1754__  
> gpt4all __2339__  
> instruct __3794__  
> Total [21251] samples and [ 7.06 ] M tokens.


#### 评测结果
| Dataset      | score |
| ----------- | ----------- |
| score      | 1.644257       |
| Reasoning   | 0.979592        |
| Common Sense      | 1.000000      |
| Truthfulness   | 1.344694        |
| Math      | 1.619048       |
| English Knowledge   | 0.994500        |
| Chinese Knowledge   | 1.003039        |
| Summarization      | 4.568930       |

#### 目录结构
--analysis 使用 data-juicer 对 input 的分析结果  
--filtered_data 普通筛选后的 HC3_ChatGPT 和 instruct  
--filtered_mathdata 数学筛选后的 gpt4all 和 instruct  
--output  
&nbsp;&nbsp;--evals 初赛评估结果  
&nbsp;&nbsp;--lora_model 模型权重  
&nbsp;&nbsp;--sft_data get_mixture 混合结果  
--solution  
&nbsp;&nbsp;--configs 配置文件
&nbsp;&nbsp;--math_configs 筛选数学的配置文件  
--toolkit  
&nbsp;&nbsp;--data-juicer  
&nbsp;&nbsp;--evaluation  
&nbsp;&nbsp;--training  
#### 注：  
如需完整目录结构的相关代码，可联系 <a href="https://github.com/bebetterest">bebetterest</a>, <a href="https://github.com/parallelsucc">parallelsucc</a> 通过阿里云盘快传分享。  
