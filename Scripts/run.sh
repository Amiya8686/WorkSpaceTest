CUDA_VISIBLE_DEVICES=0 python run.py \
--model ./model_cache/Meta-Llama-3.1-8B-Instruct \
--use_commonKV  --commonKV_parameters ./SVD_KV_parameters/Meta-Llama-3.1-8B-Instruct/g4_e0.9_b0_e32/commonKV_parameters.safetensors \
--use_let --let_parameters ./SVD_KV_parameters/Meta-Llama-3.1-8B-Instruct/g4_e0.9_b0_e32/let_parameters.safetensors \
--log_path ./logs/Meta-Llama-3.1-8B-Instruct/g4_e0.9_b0_e32 \
--train_let --nsamples 20 --epoches 20 --loss_scale_factor 10 \
--let_save_dir ./SVD_KV_parameters/trained_let_parameters \
--eval_latentKV_similarity \








