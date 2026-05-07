CUDA_VISIBLE_DEVICES=0 python ./ChangeModelToCommonKV/SVD_KV_parameters.py \
--model ./model_cache/Meta-Llama-3.1-8B-Instruct \
--save_dir ./SVD_KV_parameters/Meta-Llama-3.1-8B-Instruct \
--begin_layer 0 --end_layer 32 --group_num 4 --energy_threshold 0.9


