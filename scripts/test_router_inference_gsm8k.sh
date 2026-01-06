export OPENAI_API_KEY=""
export OPENAI_BASE_URL="https://api.shubiaobiao.cn/v1/"

experiment_name="gsm8k_router_inference_gpt-4.1-nano_gpt-4o-mini_deepseek-v3"
router_path="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/router_models/gpt-4.1-nano_gpt-4o-mini_deepseek-v3_lr_1e-3/checkpoints/epoch_0.pt"

# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client experiments/run_gsm8k.py \
# --mode FullConnected \
# --llm_name qwen3-32b \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --to_graph_dir $graph_dir \
# --experiment_name $experiment_name \

python experiments/run_router_gsm8k.py \
--mode FullConnected \
--llm_name gpt-4o-mini \
--batch_size 4 \
--agent_nums 4 \
--num_iterations 10 \
--num_rounds 1 \
--optimized_spatial \
--experiment_name $experiment_name \
--use_checkpoint