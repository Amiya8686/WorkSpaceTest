CUDA_VISIBLE_DEVICES=0 python run.py \
--log_path logs/Llama3.1-8B-instruct/origin/ \
--eval_my_text --eval_hidden_state_similarity  --eval_KVcache_similarity



CUDA_VISIBLE_DEVICES=0 python run.py \
--use_commonKV \
--log_path logs/Llama3.1-8B-instruct/g4e0.9/ \
--eval_my_text --eval_hidden_state_similarity  --eval_latentKV_similarity


