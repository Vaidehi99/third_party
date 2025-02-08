# Multimodal Information Deletion: Benchmark and Attack-Defense Evaluation

This repository includes code for the paper:

[Unlearning Sensitive Information in Multimodal LLMs: Benchmark and Attack-Defense Evaluation](https://openreview.net/forum?id=YcnjgKbZQS)

[Vaidehi Patil](https://vaidehi99.github.io/),  [Yi-Lin Sung](https://ylsung.github.io/) ,[Peter Hase](https://peterbhase.github.io/), Jie Peng, [Tianlong Chen](https://tianlong-chen.github.io/) and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)


![image](./assets/overall.jpg)

*Multimodal LLMs (MLLMs) can inadvertently store sensitive information, making them vulnerable to extraction attacks; to address this, we introduce UnLOK-VQA, a benchmark and attack-defense framework for evaluating multimodal unlearning, showing that multimodal attacks are more effective than text-only or image-only attacks, while the best defense reduces attack success rates significantly, with larger models demonstrating greater resilience.*

## Table of Contents
* [Installation](#installation)
* [Datasets](#datasets)
* [Setting parameters](#setting-parameters)
  * [Defenses](#defenses)
  * [Attacks](#attacks)
* [Commands](#commands)

## Installation

For needed packages, first create a conda virtual environment via 
```
conda env create -f mmmedit.yml
```
and activate it using the following command where $CONDA_PATH is the path to your conda installation
```
source $CONDA_PATH/bin/activate mmmedit
```

Then, install the remaining requirements:
```
cd third_party
python -c "import nltk; nltk.download('punkt')"
```


cd /nas-ssd2/vaidehi/nlp13/belief-localization/third_party
source /nas-ssd2/vaidehi/projects/anaconda3/etc/profile.d/conda.sh
conda activate MMMEdit_rebuttal
export PYTHONPATH=/nas-ssd2/vaidehi/projects/LLaVA/cache
export HF_HOME=cache/

CUDA_VISIBLE_DEVICES="7" python3 -m experiments.evaluate_llava_mm_parap     -n 10     --alg_name FT     --window_sizes "1"     --ds_name zsre     --model_name liuhaotian/llava-v1.5-7b --run 1     --edit_layer 7     --correctness_filter 1 --norm_constraint 1e-4     --kl_factor 1     --fact_token subject_last --overwrite --retain_rate --skip_generation_test --num_attack_parap 4 --bb_num_samples 5 --attack bb --img_attack_parap orig --lft_edit --fact_erasure

CUDA_VISIBLE_DEVICES="3" python3 -m experiments.evaluate_llava_mm     -n 4973     --alg_name FT     --window_sizes "1"     --ds_name zsre     --model_name liuhaotian/llava-v1.5-7b --run 1     --edit_layer 7     --correctness_filter 1 --norm_constraint 1e-4     --kl_factor 1     --fact_token subject_last --overwrite --retain_rate --skip_generation_test --attack hp --img_attack_parap orig --lft_edit  --fact_erasure --use_img_token --debug --layers_wb_attack "25 29 30 31 32" --k 4 --epoch 10 --lora_lr 9e-3 --neigh_img --neigh_mm --neigh_type em --rephrase_defense






CUDA_VISIBLE_DEVICES="2" python3 -m experiments.evaluate_llava_mm     -n 10     --alg_name FT     --window_sizes "1"     --ds_name zsre     --model_name liuhaotian/llava-v1.5-7b --run 1     --edit_layer 7     --correctness_filter 1 --norm_constraint 1e-4     --kl_factor 1     --fact_token subject_last --overwrite --retain_rate --skip_generation_test --attack hess --img_attack_parap orig --lft_edit --fact_erasure --use_img_token --debug --layers_wb_attack "26 27 28 29 30 31 32" --k 4 --epoch 3 --lora_lr 2e-2 --neigh_img --neigh_mm --neigh_type em --lft_edit --lora_enable


Full FT: Add  --lora_enable

CUDA_VISIBLE_DEVICES="2" python3 -m experiments.evaluate_llava_mm     -n 10     --alg_name FT     --window_sizes "1"     --ds_name zsre     --model_name liuhaotian/llava-v1.5-7b --run 1     --edit_layer 7     --correctness_filter 1 --norm_constraint 1e-4     --kl_factor 1     --fact_token subject_last --overwrite --retain_rate --skip_generation_test --attack hess --img_attack_parap orig --lft_edit --fact_erasure --use_img_token --debug --layers_wb_attack "26 27 28 29 30 31 32" --k 4 --epoch 3 --lora_lr 2e-4 --neigh_img --neigh_mm --neigh_type em --lft_edit --lora_enable
