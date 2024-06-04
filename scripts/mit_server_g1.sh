###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-04-04 21:33:17
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-06-04 21:04:23
 # @FilePath: /third_party/scripts/mit_server_g1.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 


export TRANSFORMERS_CACHE=/data1/tianlong/cache/
export HOME=/data1/tianlong


gpu_id=3

layers_wb_attack="22,23,24,25,26,27,28,29,30,31,32"

# args=" 
#     -n 600
#     --alg_name FT
#     --window_sizes 1
#     --ds_name zsre
#     --model_name liuhaotian/llava-v1.5-7b
#     --run 1
#     --edit_layer 7
#     --correctness_filter 1
#     --norm_constraint 1e-4
#     --kl_factor 1
#     --fact_token subject_last
#     --fact_erasure
#     --overwrite
#     --attack pd
#     --layers_wb_attack ${layers_wb_attack}
#     --k 2
#     --retain_rate
#     --skip_generation_tests 
#     --fact_erasure 
#     --debug 
#     --use_img_token
#     --img_attack_parap orig 
#     --window_sizes 1 
#     --cft_edit
#     --skip_generation_tests
# "

# layers_wb_attack="25,29,30,31,32"
layers_wb_attack="26,27,28,29,30,31,32"
margin_layers="30 31 32 33 34 35 36 37 38 39 40"



linx="9"
lrs="1e-2 1e-3 5e-3 1e-4"
epochs="10 15"


args=" 
    -n 5000 
    --alg_name FT
    --window_sizes 1
    --ds_name zsre
    --model_name liuhaotian/llava-v1.5-13b
    --run 1
    --edit_layer 7
    --correctness_filter 1
    --norm_constraint 1e-4
    --kl_factor 1
    --fact_token subject_last
    --overwrite 
    --retain_rate 
    --skip_generation_tests 
    --num_attack_parap 4
    --bb_num_samples 5 
    --attack jailbreak 
    --img_attack_parap medium_only 
    --lft_edit 
    --fact_erasure
    --use_img_token
    --epoch 15
    --lora_lr 5e-2
"

CUDA_VISIBLE_DEVICES="1" python -m experiments.evaluate_llava_mm_parap ${args}