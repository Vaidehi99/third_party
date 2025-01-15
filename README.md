

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
