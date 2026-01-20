import sys, os
import sys
import os
from pathlib import Path

# Get the parent directory of experiments/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from datasets.mmlu_dataset import MMLUDataset
sys.stdout.reconfigure(encoding='utf-8')

import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import Union, Literal, List
import copy
import argparse
import random
import json

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from GDesigner.utils.const import GDesigner_ROOT
from experiments.utils import get_optimizer

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on MMLU")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations. Default 10.")
    parser.add_argument('--imp_per_iterations', type=int, default=5,
                        help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=1,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini",
                        help="Model name, None runs the default gpt-4o-mini")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--gumbel_tau', type=float, default=0.5,
                        help="Gumbel-Softmax temperature for edge sampling.")
    parser.add_argument('--refine_rank', type=int, default=None,
                        help="Rank used in the refine module (default uses full rank).")
    parser.add_argument('--refine_zeta', type=float, default=1e-3,
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
    parser.add_argument('--dataset_start_index', type=int, default=0,
                        help="Start index of the dataset.")
    parser.add_argument('--num_of_data', type=int, default=None,
                        help="Number of data to run.")

    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
        
    return args

async def main():
    args = parse_args()
    result_file = None
    
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
        result_dir = Path(f"{GDesigner_ROOT}/result/mmlu")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"
        if result_file.exists():
            data = load_result(result_file)
            executed_batch = len(data)/args.batch_size
            last_result = data[-1]
            total_solved, total_executed = last_result["Total Solved"], last_result["Total executed"]
        else:
            total_solved, total_executed = (0, 0)

    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode,len(agent_names))
    limit_questions = 153
    
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  gumbel_tau=args.gumbel_tau,
                  refine_rank=args.refine_rank,
                  refine_zeta=args.refine_zeta,
                  **kwargs)
    download()
    # dataset_train = MMLUDataset('dev')
    # dataset_val = MMLUDataset('val')
    dataset = MMLUDataset('dev')
    dataset._total_df = dataset._total_df.iloc[args.dataset_start_index:]

    # if args.optimized_spatial or args.optimized_temporal:
    #     await train(graph=graph,dataset=dataset_train,num_iters=args.num_iterations,num_rounds=args.num_rounds,
    #                 lr=args.lr,batch_size=args.batch_size,
    #                 anchor_weight=args.anchor_weight,sparse_weight=args.sparse_weight)
    num_iters = args.num_iterations
    num_rounds = args.num_rounds
    # if args.optimized_spatial:
    #     optimizer, trainable_models = get_optimizer(graph, lr=args.lr)
    args.optimized_spatial = False
    args.optimized_temporal = False
    graph.eval()
    print("Start Eval")
    batch_size = args.batch_size
    anchor_weight = args.anchor_weight
    sparse_weight = args.sparse_weight
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    
    # trainable_params = []
    # trainable_params += list(graph.gcn.parameters())
    # trainable_params += list(graph.mlp.parameters())
    # trainable_params += list(graph.encoder_mu.parameters())
    # trainable_params += list(graph.encoder_logvar.parameters())
    # trainable_params += list(graph.qs_linear.parameters())
    # trainable_params += list(graph.refine.parameters())
    # optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # graph.gcn.train()
    # graph.mlp.train()
    # graph.encoder_mu.train()
    # graph.encoder_logvar.train()
    # graph.qs_linear.train()
    # graph.refine.train()
    device = next(graph.gcn.parameters()).device

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80*'-')
        start_ts = time.time()
        tasks = []
        correct_answers = []
        answer_log_probs = []
        realized_graphs: List[Graph] = []

        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            realized_graph.encoder_mu = graph.encoder_mu
            realized_graph.encoder_logvar = graph.encoder_logvar
            # realized_graph.qs_linear = graph.qs_linear
            realized_graph.refine = graph.refine
            realized_graphs.append(realized_graph)
            input_dict = dataset.record_to_input(record)
            print("Input dict", 80*'=')
            print(input_dict)
            tasks.append(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)
        answers: List[str] = []
        anchor_losses: List[torch.Tensor] = []
        sparse_losses: List[torch.Tensor] = []

        for realized_graph, task, raw_answer, log_prob, correct_answer in zip(realized_graphs, tasks, raw_answers, log_probs, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            total_solved += accuracy._num_correct
            total_executed += accuracy._num_total
            accuracy_float = total_solved / total_executed
            
            utility = accuracy.get()
            is_solved = True if utility == 1 else False
            utilities.append(utility)
            single_loss = - log_prob * utility
            loss_list.append(single_loss)
            print(f"correct answer:{correct_answer}")

            if hasattr(realized_graph, "refine_losses"):
                anchor_losses.append(realized_graph.refine_losses.get("L_anchor", torch.tensor(0.0, device=device)).to(device))
                sparse_losses.append(realized_graph.refine_losses.get("L_sparse", torch.tensor(0.0, device=device)).to(device))

            updated_item = {
                "Question": task,
                "Tests": correct_answer,
                "Attempt answer": raw_answer,
                "Solved": is_solved,
                "Solution": raw_answer,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy_float,
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

        L_utility = torch.mean(torch.stack(loss_list)) if loss_list else torch.tensor(0.0, device=device)
        L_anchor = torch.mean(torch.stack(anchor_losses)) if anchor_losses else torch.tensor(0.0, device=device)
        L_sparse = torch.mean(torch.stack(sparse_losses)) if sparse_losses else torch.tensor(0.0, device=device)
        L_GDesigner = L_utility + anchor_weight * L_anchor + sparse_weight * L_sparse
        # optimizer.zero_grad()
        # if args.optimized_spatial or args.optimized_temporal:
        #     optimizer.zero_grad()
        #     L_GDesigner.backward()
        #     for model_name, model in trainable_models.items():
        #         for name, p in model.named_parameters():
        #             if p.grad is not None and not torch.isfinite(p.grad).all():
        #                 print(model_name)
        #                 print(f"[NaNGuard] {name}.grad has NaN/Inf")
        #     optimizer.step()

        print("raw_answers:",raw_answers)
        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities) # [0.0, 0.0, 0.0, 1.0]
        print(f"Accuracy: {accuracy_float}")
        print("L_utility:", L_utility.item())
        print("L_anchor:", L_anchor.item())
        print("L_sparse:", L_sparse.item())
        print("L_GDesigner:", L_GDesigner.item())
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

        # if total_executed >= 40:  # Train on 40
        #     if args.to_graph_dir:
        #         graph.save_graph(args.to_graph_dir)
        #     break
        if args.num_of_data and total_executed >= args.num_of_data:
            break
        
def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['Mesh'],
                          Literal['FakeFullConnected'],Literal['FakeRandom'],Literal['FakeChain'],Literal['FakeStar'],Literal['FakeMesh'],Literal['FakeAGRandom'],Literal['FakeAGFull']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0]*N for _ in range(N)]
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
    
    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i+1,N):
                adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1,N):
            adj_matrix[0][i] = 1
        return adj_matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Normal'}]
    elif mode=='FullConnected' or mode == 'FakeFullConnected' or mode=='FakeAGFull':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh' or mode=='FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star' or mode=='FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':None} for i in range(N)]
        
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == "__main__":
    asyncio.run(main())