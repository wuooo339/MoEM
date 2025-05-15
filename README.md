# MoE-Modified
This code is originated from ​my Graduation Project​.If you want to know more information, contact me at 1092897051@qq.com
## Installation
We recommend installing in a virtual environment of python=3.9
```bash
conda create -n myenv python=3.9
conda activate myenv
# install from either PyPI or Source will trigger requirements.txt automatically
```
## Usage and Examples
All is the same with moe-infinity but add some techniques to deal with requiest lever expert cache and expert prefetch information.

use the following command to run:
```shell
CUDA_VISIBLE_DEVICES=0 python run.py 2>&1 | tee output/log_test.txt
```
## Prefetch 
if you want to use prefetch mode, find line 101 in deepseek.py, that part is ​commented out​.
```python
expert_index = topk_idx.reshape(batch_size, sequence_length, self.config.
print(f"prefill = {GLOBAL_CONFIG} ,layer = {self.layer_id},expert_id = {expert_index}")
if GLOBAL_CONFIG["prefill"] == 1 and self.layer_id > 25:
    ...
```
## Conference
The repository is built upon the moe-infinity[paper](https://arxiv.org/abs/2401.14361):
