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

## Dataset: UnLOK-VQA (Unlearning Outside Knowledge VQA)
Link: 
The dataset has ~500 entries, each with the following keys:
"id": ID 
"src": Question whose answer is to be deleted
"pred": Answer to the question which is meant to be deleted
"loc": Neighborhood questions
"loc_ans": Answers to the neighborhood questions,
"image_id": Image ID of the image in COCO dataset

To get the Neighborhood Images (Easy, Hard) and Paraphrase Images (Easy, Medium, Hard), see this link.

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
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```
## Setting parameters

### Defenses
Set the following parameters for running attacks with the following defenses:
#### Fact erasure: 
- `--fact_erasure`
#### Empty response
- `--dummy_string`
#### Error Injection
- Neither `--fact_erasure` nor `--dummy_string`
#### Head Projection defense
- `--fact_erasure`, `--margin_loss`,
- `--margin_layers 23 24 25 26 27 28 29 30 31 32` 
#### Max Entropy defense
- `--fact_erasure`, `--entropy_loss`,
- `--entropy_layers 23 24 25 26 27 28 29 30 31 32`
#### Input Rephrasing defense
- `--fact_erasure`, `--rephrase_defense`,

## Commands



Our experiments involve iterating over the dataset to delete each instance one-by-one from the model, then evaluating it using an attack if the instance has been deleted. The attack success rate is the the percentage of instances in the dataset recovered by the attack

Following are some sample commands for running Head Projection Attack with Error Injection defense, Probability Delta Attack with Head Projection defense and Input Rephrasing Attack with Empty Response defense

Evaluate Fact Erasure defense against the Question Rephrase Attack

```
CUDA_VISIBLE_DEVICES="7" python3 -m experiments.evaluate_llava_mm_parap     -n 10     --alg_name FT     --window_sizes "1"     --ds_name zsre     --model_name liuhaotian/llava-v1.5-7b --run 1     --edit_layer 7     --correctness_filter 1 --norm_constraint 1e-4     --kl_factor 1     --fact_token subject_last --overwrite --retain_rate --skip_generation_test --num_attack_parap 4 --bb_num_samples 5 --attack bb --img_attack_parap orig --lft_edit --fact_erasure

```

Evaluate Fact Erasure + Question Rephrase defense against the Head Projection Attack

```
CUDA_VISIBLE_DEVICES="3" python3 -m experiments.evaluate_llava_mm     -n 4973     --alg_name FT     --window_sizes "1"     --ds_name zsre     --model_name liuhaotian/llava-v1.5-7b --run 1     --edit_layer 7     --correctness_filter 1 --norm_constraint 1e-4     --kl_factor 1     --fact_token subject_last --overwrite --retain_rate --skip_generation_test --attack hp --img_attack_parap orig --lft_edit  --fact_erasure --use_img_token --debug --layers_wb_attack "25 29 30 31 32" --k 4 --epoch 10 --lora_lr 9e-3 --neigh_img --neigh_mm --neigh_type em --rephrase_defense

```

Evaluate Error Injection defense against the PD^2^ Attack


```
CUDA_VISIBLE_DEVICES="2" python3 -m experiments.evaluate_llava_mm     -n 10     --alg_name FT     --window_sizes "1"     --ds_name zsre     --model_name liuhaotian/llava-v1.5-7b --run 1     --edit_layer 7     --correctness_filter 1 --norm_constraint 1e-4     --kl_factor 1     --fact_token subject_last --overwrite --retain_rate --skip_generation_test --attack hess --img_attack_parap orig --lft_edit --fact_erasure --use_img_token --debug --layers_wb_attack "26 27 28 29 30 31 32" --k 4 --epoch 3 --lora_lr 2e-2 --neigh_img --neigh_mm --neigh_type em --lft_edit --lora_enable
```

To run full fine-tuning, add ```--lora_enable``` to the above commands:


```
Full FT: Add  --lora_enable

CUDA_VISIBLE_DEVICES="2" python3 -m experiments.evaluate_llava_mm     -n 10     --alg_name FT     --window_sizes "1"     --ds_name zsre     --model_name liuhaotian/llava-v1.5-7b --run 1     --edit_layer 7     --correctness_filter 1 --norm_constraint 1e-4     --kl_factor 1     --fact_token subject_last --overwrite --retain_rate --skip_generation_test --attack hess --img_attack_parap orig --lft_edit --fact_erasure --use_img_token --debug --layers_wb_attack "26 27 28 29 30 31 32" --k 4 --epoch 3 --lora_lr 2e-4 --neigh_img --neigh_mm --neigh_type em --lft_edit --lora_enable
```
