import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.tools.coding.python_executor import PyExecutor
from GDesigner.utils.globals import Time
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from experiments.utils import get_optimizer

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str, default="datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--num_rounds',type=int,default=2,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default = 10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decison method of the GDesigner')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    
    # New arguments for training
    parser.add_argument('--gumbel_tau', type=float, default=1,
                        help="Gumbel-Softmax temperature for edge sampling.")
    parser.add_argument('--refine_rank', type=int, default=2,
                        help="Rank used in the refine module (default uses full rank).")
    parser.add_argument('--refine_zeta', type=float, default=1e-1,
                        help="Nuclear norm regularization strength for the refine module.")
    parser.add_argument('--anchor_weight', type=float, default=1.0,
                        help="Weight of the anchor loss during training.")
    parser.add_argument('--sparse_weight', type=float, default=1.0,
                        help="Weight of the sparse regularization loss during training.")
    parser.add_argument('--from_graph_dir', type=str, default=None,
                        help="Directory to load the graph.")
    parser.add_argument('--to_graph_dir', type=str, default=None,
                        help="Directory to save the graph.")
    parser.add_argument('--experiment_name', type=str, default=None,
                        help="Name of the experiment.")
    
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

async def main():
    args = parse_args()
    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    # Result file and resume logic
    executed_batch = None
    if args.experiment_name:
        experiment_dir = Path(f"{GDesigner_ROOT}/result/{args.experiment_name}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        result_dir = experiment_dir / f"{args.domain}_{args.llm_name}"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"result.json"
        if result_file.exists():
            data = load_result(result_file)
            executed_batch = len(data)/args.batch_size
            last_result = data[-1]
            total_solved, total_executed = last_result["Total solved"], last_result["Total executed"]
            cost = last_result["cost"]
            prompt_tokens = last_result["prompt_tokens"]
            completion_tokens = last_result["completion_tokens"]
            Cost.instance().value = cost
            PromptTokens.instance().value = prompt_tokens
            CompletionTokens.instance().value = completion_tokens
        else:
            total_solved, total_executed = (0, 0)
    else:
        result_dir = Path(f"{GDesigner_ROOT}/result/humaneval")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"
        if result_file.exists():
            data = load_result(result_file)
            executed_batch = len(data)/args.batch_size
            last_result = data[-1]
            total_solved, total_executed = last_result["Total Solved"], last_result["Total executed"]
        else:
            total_solved, total_executed = (0, 0)
    
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))
    
    if args.from_graph_dir and executed_batch:
        graph = Graph.load_graph(
            graph_dir = args.from_graph_dir,
            domain="humaneval",
            llm_name=args.llm_name,
            agent_names=agent_names,
            decision_method=decision_method,
            optimized_spatial=args.optimized_spatial,
            optimized_temporal=args.optimized_temporal,
            gumbel_tau=args.gumbel_tau,
            refine_rank=args.refine_rank,
            refine_zeta=args.refine_zeta,
            **kwargs
        )
    else:
        graph = Graph(domain="humaneval",
                    llm_name=args.llm_name,
                    agent_names=agent_names,
                    decision_method=decision_method,
                    optimized_spatial=args.optimized_spatial,
                    optimized_temporal=args.optimized_temporal,
                    gumbel_tau=args.gumbel_tau,
                    refine_rank=args.refine_rank,
                    refine_zeta=args.refine_zeta,
                    **kwargs)
    device = next(graph.gcn.parameters()).device
    print(f"Device: {device}")

    # graph.gcn.train()
    # optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=args.lr) 
    if args.optimized_spatial:
        optimizer, trainable_models = get_optimizer(graph, lr=args.lr)

    anchor_weight = args.anchor_weight
    sparse_weight = args.sparse_weight   
    
    num_batches = int(len(dataset)/args.batch_size)
    # total_solved, total_executed = (0, 0)
    for i_batch in range(num_batches):

        if executed_batch:
            if i_batch < executed_batch:
                if i_batch+1 == args.num_iterations:
                    args.optimized_spatial = False
                    args.optimized_temporal = False
                    total_solved = 0
                    total_executed = 0
                    graph.eval()
                    print("Start Eval")
                continue

        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        tests = []
        realized_graphs: List[Graph] = []
        
        current_batch = dataloader(dataset,args.batch_size,i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        
        for i_record, record in enumerate(current_batch):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp

            realized_graph.encoder_mu = graph.encoder_mu
            realized_graph.encoder_logvar = graph.encoder_logvar
            realized_graph.ps_linear = graph.ps_linear
            realized_graph.refine = graph.refine
            realized_graphs.append(realized_graph)

            task = record["prompt"]
            test = record["test"]
            tests.append(test)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)
        anchor_losses: List[torch.Tensor] = []
        sparse_losses: List[torch.Tensor] = []
        
        for realized_graph, task, answer, log_prob, test in zip(realized_graphs, current_batch, raw_answers, log_probs, tests):
            if not isinstance(answer,list):
                raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
            answer = answer[0].lstrip("```python\n").rstrip("\n```")
            is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved / total_executed

            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)

            if hasattr(realized_graph, "refine_losses"):
                anchor_losses.append(realized_graph.refine_losses.get("L_anchor", torch.tensor(0.0, device=device)).to(device))
                sparse_losses.append(realized_graph.refine_losses.get("L_sparse", torch.tensor(0.0, device=device)).to(device))

            updated_item = {
                "Question": task,
                "Tests": test,
                "Attempt answer": answer,
                "Solved": is_solved,
                "Solution": answer,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy,
                "utility": utility,
                "anchor_loss": anchor_losses[-1].item(),
                "sparse_loss": sparse_losses[-1].item(),
                "cost": Cost.instance().value,
                "prompt_tokens": PromptTokens.instance().value,
                "completion_tokens": CompletionTokens.instance().value,
            }
            data.append(updated_item)
            realized_graph.save_result(result_dir, str(total_executed))
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        # total_loss = torch.mean(torch.stack(loss_list))
        # if args.optimized_spatial or args.optimized_temporal:
        #     optimizer.zero_grad()
        #     total_loss.backward()
        #     optimizer.step()
        # print(f"Batch time {time.time() - start_ts:.3f}")
        # print(f"Accuracy: {accuracy}")
        # print("utilities:", utilities)
        # print("loss:", total_loss.item())

        L_utility = torch.mean(torch.stack(loss_list)) if loss_list else torch.tensor(0.0, device=device)
        L_anchor = torch.mean(torch.stack(anchor_losses)) if anchor_losses else torch.tensor(0.0, device=device)
        L_sparse = torch.mean(torch.stack(sparse_losses)) if sparse_losses else torch.tensor(0.0, device=device)
        L_GDesigner = L_utility + anchor_weight * L_anchor + sparse_weight * L_sparse
        if not torch.isfinite(L_GDesigner):
            print(f"[NaNGuard] L_GDesigner has NaN/Inf")
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            L_GDesigner.backward()
            for model_name, model in trainable_models.items():
                for name, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(model_name)
                        print(f"[NaNGuard] {name}.grad has NaN/Inf")
            optimizer.step()
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)
        print("loss:", L_GDesigner.item())
        print("L_utility:", L_utility.item())
        print("L_anchor:", L_anchor.item())
        print("L_sparse:", L_sparse.item())
        print("L_GDesigner:", L_GDesigner.item())

        if i_batch+1 == args.num_iterations:
            if args.to_graph_dir:
                graph.save_graph(args.to_graph_dir)
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = 0
            total_executed = 0
            graph.eval()
            print("Start Eval")
            break
            
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

        if total_executed >= 40:  # Train on 40
            break


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix

    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    asyncio.run(main())
