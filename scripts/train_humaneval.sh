export OPENAI_API_KEY="sk-IlhmAWpQFIfc5a0IF566F7Fe93A04522A255422c68158fD7"
export OPENAI_BASE_URL="https://api.shubiaobiao.cn/v1/"
experiment_name="train_humaneval_gpt4o-mini"
graph_dir="/workspace/ziqi/GRouter/graph_result/$experiment_name"

nohup python experiments/train_humaneval.py \
--mode FullConnected \
--llm_name gpt-4o-mini \
--batch_size 4 \
--agent_nums 5 \
--num_iterations 10 \
--num_rounds 2 \
--optimized_spatial \
--to_graph_dir $graph_dir \
--experiment_name $experiment_name \

# experiment_name="train_gsm8k_gpt4.1-nano"
# graph_dir="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/$experiment_name"

# python experiments/train_gsm8k.py \
# --mode FullConnected \
# --llm_name gpt-4.1-nano \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --to_graph_dir $graph_dir \
# --experiment_name $experiment_name \

# experiment_name="train_gsm8k_gpt-5-nano"
# graph_dir="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/$experiment_name"

# python experiments/train_gsm8k.py \
# --mode FullConnected \
# --llm_name gpt-5-nano \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --to_graph_dir $graph_dir \
# --experiment_name $experiment_name \

# experiment_name="train_gsm8k_qwen3-32b"
# graph_dir="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/$experiment_name"

# # # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client experiments/run_gsm8k.py \
# # # --mode FullConnected \
# # # --llm_name qwen3-32b \
# # # --batch_size 4 \
# # # --agent_nums 4 \
# # # --num_iterations 10 \
# # # --num_rounds 1 \
# # # --optimized_spatial \
# # # --to_graph_dir $graph_dir \
# # # --experiment_name $experiment_name \

# python experiments/train_gsm8k.py \
# --mode FullConnected \
# --llm_name qwen3-32b \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --to_graph_dir $graph_dir \
# --experiment_name $experiment_name \

# experiment_name="train_gsm8k_deepseek-v3"
# graph_dir="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/$experiment_name"

# python experiments/train_gsm8k.py \
# --mode FullConnected \
# --llm_name deepseek-v3 \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --to_graph_dir $graph_dir \
# --experiment_name $experiment_name \