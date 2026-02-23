export OPENAI_API_KEY="sk-IlhmAWpQFIfc5a0IF566F7Fe93A04522A255422c68158fD7"
export OPENAI_BASE_URL="https://api.shubiaobiao.cn/v1/"
experiment_name="humaneval_gpt4o-mini"
graph_dir="/workspace/ziqi/GRouter/graph_result/train_humaneval_gpt4o-mini"
dataset_json="datasets/humaneval/humaneval-py.jsonl"
dataset_start_index=0
num_of_data=40

nohup python experiments/run_humaneval.py \
--mode FullConnected \
--llm_name gpt-4o-mini \
--dataset_json $dataset_json \
--batch_size 4 \
--agent_nums 5 \
--num_iterations 10 \
--num_rounds 2 \
--optimized_spatial \
--from_graph_dir $graph_dir \
--experiment_name $experiment_name \
--dataset_start_index $dataset_start_index \
--num_of_data $num_of_data \

# experiment_name="humaneval_gpt-4.1-nano"
# graph_dir="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_humaneval_gpt4.1-nano"

# python experiments/run_humaneval.py \
# --mode FullConnected \
# --llm_name gpt-4.1-nano \
# --dataset_json $dataset_json \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --from_graph_dir $graph_dir \
# --experiment_name $experiment_name \
# --dataset_start_index $dataset_start_index \
# --num_of_data $num_of_data \

# experiment_name="humaneval_gpt-5-nano"
# graph_dir="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_humaneval_gpt-5-nano"

# python experiments/run_humaneval.py \
# --mode FullConnected \
# --llm_name gpt-5-nano \
# --dataset_json $dataset_json \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --from_graph_dir $graph_dir \
# --experiment_name $experiment_name \
# --dataset_start_index $dataset_start_index \
# --num_of_data $num_of_data \


# experiment_name="humaneval_qwen3-32b"
# graph_dir="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_humaneval_qwen3-32b"

# # # python -m debugpy --listen 0.0.0.0:5678 --wait-for-client experiments/run_humaneval.py \
# # # --mode FullConnected \
# # # --llm_name qwen3-32b \
# # # --batch_size 4 \
# # # --agent_nums 4 \
# # # --num_iterations 10 \
# # # --num_rounds 1 \
# # # --optimized_spatial \
# # # --to_graph_dir $graph_dir \
# # # --experiment_name $experiment_name \

# python experiments/run_humaneval.py \
# --mode FullConnected \
# --llm_name qwen3-32b \
# --dataset_json $dataset_json \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --from_graph_dir $graph_dir \
# --experiment_name $experiment_name \
# --dataset_start_index $dataset_start_index \
# --num_of_data $num_of_data \

# experiment_name="humaneval_deepseek-v3"
# graph_dir="/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_humaneval_deepseek-v3"

# python experiments/run_humaneval.py \
# --mode FullConnected \
# --llm_name deepseek-v3 \
# --dataset_json $dataset_json \
# --batch_size 4 \
# --agent_nums 4 \
# --num_iterations 10 \
# --num_rounds 1 \
# --optimized_spatial \
# --from_graph_dir $graph_dir \
# --experiment_name $experiment_name \
# --dataset_start_index $dataset_start_index \
# --num_of_data $num_of_data \