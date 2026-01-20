import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens

async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            anchor_weight: float = 1.0,
            sparse_weight: float = 1.0,
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    
    trainable_params = []
    trainable_params += list(graph.gcn.parameters())
    trainable_params += list(graph.mlp.parameters())
    trainable_params += list(graph.encoder_mu.parameters())
    trainable_params += list(graph.encoder_logvar.parameters())
    trainable_params += list(graph.qs_linear.parameters())
    trainable_params += list(graph.refine.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    graph.gcn.train()
    graph.mlp.train()
    graph.encoder_mu.train()
    graph.encoder_logvar.train()
    graph.qs_linear.train()
    graph.refine.train()
    device = next(graph.gcn.parameters()).device

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        realized_graphs: List[Graph] = []

        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            realized_graph.encoder_mu = graph.encoder_mu
            realized_graph.encoder_logvar = graph.encoder_logvar
            realized_graph.qs_linear = graph.qs_linear
            realized_graph.refine = graph.refine
            realized_graphs.append(realized_graph)
            input_dict = dataset.record_to_input(record)
            print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        anchor_losses: List[torch.Tensor] = []
        sparse_losses: List[torch.Tensor] = []

        for realized_graph, raw_answer, log_prob, correct_answer in zip(realized_graphs, raw_answers, log_probs, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            single_loss = - log_prob * utility
            loss_list.append(single_loss)
            print(f"correct answer:{correct_answer}")

            if hasattr(realized_graph, "refine_losses"):
                anchor_losses.append(realized_graph.refine_losses.get("L_anchor", torch.tensor(0.0, device=device)).to(device))
                sparse_losses.append(realized_graph.refine_losses.get("L_sparse", torch.tensor(0.0, device=device)).to(device))

        L_utility = torch.mean(torch.stack(loss_list)) if loss_list else torch.tensor(0.0, device=device)
        L_anchor = torch.mean(torch.stack(anchor_losses)) if anchor_losses else torch.tensor(0.0, device=device)
        L_sparse = torch.mean(torch.stack(sparse_losses)) if sparse_losses else torch.tensor(0.0, device=device)
        L_GDesigner = L_utility + anchor_weight * L_anchor + sparse_weight * L_sparse
        optimizer.zero_grad()
        L_GDesigner.backward()
        optimizer.step()

        print("raw_answers:",raw_answers)
        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities) # [0.0, 0.0, 0.0, 1.0]
        print("L_utility:", L_utility.item())
        print("L_anchor:", L_anchor.item())
        print("L_sparse:", L_sparse.item())
        print("L_GDesigner:", L_GDesigner.item())
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
        