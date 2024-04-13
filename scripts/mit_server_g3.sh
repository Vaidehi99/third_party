###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-04-04 21:33:17
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-04-14 00:42:17
 # @FilePath: /third_party/scripts/mit_server.sh
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
layers_wb_attack="36,37,38,39,40"
margin_layers="22 23 24 25 26 27 28 29 30 31 32"



linx="9 14 19 24 28"
lrs="5e-2"
epochs="10 20 30"

for ep in $epochs
do
for lr in $lrs
do
for li in $linx
do
args=" 
    -n 10
    --alg_name FT
    --window_sizes 1
    --ds_name zsre
    --model_name liuhaotian/llava-v1.5-13b
    --run 1
    --edit_layer $li
    --correctness_filter 1
    --norm_constraint 1e-4
    --kl_factor 1
    --fact_token subject_last
    --overwrite 
    --retain_rate 
    --skip_generation_tests 
    --attack hp 
    --img_attack_parap orig 
    --lft_edit 
    --fact_erasure 
    --use_img_token 
    --debug 
    --layers_wb_attack $layers_wb_attack
     --k 4 
     --epoch $ep
     --fact_erasure 
     --lora_lr $lr
     --use_img_token
"

CUDA_VISIBLE_DEVICES="0" python -m experiments.evaluate_llava_mm ${args}
done
done
done