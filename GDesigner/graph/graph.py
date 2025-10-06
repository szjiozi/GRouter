import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import asyncio

from GDesigner.graph.node import Node, TaskNode
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.gnn.gcn import GCN, MLP
from GDesigner.gnn.node_encoder import NodeEncoder
from torch_geometric.utils import dense_to_sparse, to_dense_adj

class RefineModule(nn.Module):
    def __init__(self, num_nodes: int, rank: Optional[int] = None, zeta: float = 1e-3):
        super().__init__()
        max_rank = num_nodes
        if rank is None:
            self.rank = max_rank
        else:
            self.rank = min(rank, max_rank)
        if self.rank <= 0:
            raise ValueError("RefineModule rank must be positive")
        self.zeta = zeta
        self.W = nn.Parameter(torch.zeros(self.rank, self.rank))
        self._latest_losses: Dict[str, torch.Tensor] = {
            "L_anchor": torch.tensor(0.0),
            "L_sparse": torch.tensor(0.0),
            "L_total": torch.tensor(0.0),
        }

    def _compute_z(self, sketch_adj: torch.Tensor) -> torch.Tensor:
        # torch.linalg.svd is differentiable w.r.t. sketch_adj
        u, _, _ = torch.linalg.svd(sketch_adj, full_matrices=False)
        return u[:, : self.rank]

    def forward(self, sketch_adj: torch.Tensor, anchor_adj: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        anchor_adj = anchor_adj.to(device=sketch_adj.device, dtype=torch.long)
        anchor_adj = to_dense_adj(anchor_adj).squeeze(0)[:-1, :-1]
        Z = self._compute_z(sketch_adj)
        reconstruction = Z @ self.W @ Z.transpose(0, 1)
        # reconstruction = 0.5 * (reconstruction + reconstruction.transpose(0, 1))

        sketch_residual = sketch_adj - reconstruction
        anchor_residual = anchor_adj - reconstruction

        L_sketch_recon = 0.5 * torch.norm(sketch_residual, p="fro") ** 2
        nuclear_norm = torch.linalg.svdvals(self.W).sum()
        L_sparse = self.zeta * nuclear_norm
        L_anchor = 0.5 * torch.norm(anchor_residual, p="fro") ** 2 + L_sketch_recon
        L_total = L_sparse + L_anchor

        losses = {
            "L_anchor": L_anchor,
            "L_sparse": L_sparse,
            "L_total": L_total,
            "Z": Z,
        }
        # keep a detached snapshot for logging/inspection without breaking autograd
        self._latest_losses = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in losses.items()
        }
        return reconstruction, losses

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self,
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                gumbel_tau: float = 1e-2,
                refine_rank: Optional[int] = None,
                refine_zeta: float = 1e-1,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        # self.agent_node_ids: List[str] = []
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        self.task_node = TaskNode(domain=self.domain, llm_name=self.llm_name)
        # self.virtual_task_id: Optional[str] = None
        # self.virtual_task_index: Optional[int] = None

        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        self.node_order: List[str] = list(self.nodes.keys())

        total_nodes = self.num_nodes
        num_agents = len(self.agent_names)
        if fixed_spatial_masks.numel() != total_nodes * total_nodes:
            spatial_mask_matrix = fixed_spatial_masks.view(num_agents, num_agents)
            expanded_spatial_mask = torch.ones((total_nodes, total_nodes), dtype=spatial_mask_matrix.dtype)
            expanded_spatial_mask[:num_agents, :num_agents] = spatial_mask_matrix
            expanded_spatial_mask.fill_diagonal_(0)
            fixed_spatial_masks = expanded_spatial_mask.view(-1)
        else:
            spatial_mask_matrix = fixed_spatial_masks.view(total_nodes, total_nodes)
            spatial_mask_matrix.fill_diagonal_(0)
            fixed_spatial_masks = spatial_mask_matrix.view(-1)

        if fixed_temporal_masks.numel() != total_nodes * total_nodes:
            temporal_mask_matrix = fixed_temporal_masks.view(num_agents, num_agents)
            expanded_temporal_mask = torch.ones((total_nodes, total_nodes), dtype=temporal_mask_matrix.dtype)
            expanded_temporal_mask[:num_agents, :num_agents] = temporal_mask_matrix
            fixed_temporal_masks = expanded_temporal_mask.view(-1)
        else:
            fixed_temporal_masks = fixed_temporal_masks.view(total_nodes, total_nodes).reshape(-1)

        self.prompt_set = PromptSetRegistry.get(domain)
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        # self.node_encoder = NodeEncoder(self.features.size(1), self.features.size(1))
        # self.A_anchor, self.A_anchor_edge_index = self.construct_anchor_adj_matrix()
        # self.tilde_A_anchor = role_adj_matrix
        self.hidden_dim = 32
        self.latent_dim = 16
        self.gumbel_tau = gumbel_tau

        input_dim = self.features.size(1)
        self.gcn = GCN(input_dim, self.hidden_dim, self.hidden_dim)
        self.mlp = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        self.encoder_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.ps_linear = nn.Linear(self.latent_dim*3, 1)
        self.refine = RefineModule(self.num_nodes, rank=refine_rank, zeta=refine_zeta)
        self.refine_losses: Dict[str, torch.Tensor] = {
            "L_anchor": torch.tensor(0.0),
            "L_sparse": torch.tensor(0.0),
            "L_total": torch.tensor(0.0),
        }

        self.mu: Optional[torch.Tensor] = None
        self.logvar: Optional[torch.Tensor] = None
        self.latent_z: Optional[torch.Tensor] = None
        self.tilde_S: Optional[torch.Tensor] = None

        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks
        # self.spatial_edge_probs = torch.full((len(self.potential_spatial_edges),),
        #                                      initial_spatial_probability,
        #                                      dtype=torch.float32)
        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        # self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
        #                                          requires_grad=optimized_spatial) # trainable edge logits

        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
        # self.temporal_edge_probs = torch.full((len(self.potential_temporal_edges),),
        #                                       initial_temporal_probability,
        #                                       dtype=torch.float32)
        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
    
    def construct_adj_matrix(self):
        role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes+1,num_nodes+1))
        role_2_id: Dict[str, List[int]] = {}

        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id.setdefault(role, []).append(i)

        for edge in role_connect:
            in_role,out_role = edge
            in_ids = role_2_id.get(in_role, [])
            out_ids = role_2_id.get(out_role, [])
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1
        role_adj[:-1, -1] = 1
        role_adj[-1, :-1] = 1
        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index

    def construct_features(self):
        features: List[torch.Tensor] = []
        feature_dim: Optional[int] = None

        for node_id in self.nodes:
            node = self.nodes[node_id]
            # if node_id == self.virtual_task_id:
            #     if feature_dim is None:
            #         placeholder = torch.tensor(np.array(get_sentence_embedding("virtual task")), dtype=torch.float32)
            #         feature_dim = placeholder.size(0)
            #     feature_tensor = torch.zeros(feature_dim, dtype=torch.float32)
            # else:
                # role = node.role
                # profile = self.prompt_set.get_description(role)
                # feature_array = get_sentence_embedding(profile)
                # feature_tensor = torch.tensor(np.array(feature_array), dtype=torch.float32)
                # if feature_dim is None:
                #     feature_dim = feature_tensor.size(0)
            role = node.role
            profile = self.prompt_set.get_description(role)
            feature_array = get_sentence_embedding(profile)
            feature_tensor = torch.tensor(np.array(feature_array), dtype=torch.float32)
            if feature_dim is None:
                feature_dim = feature_tensor.size(0)
            features.append(feature_tensor)

        if not features:
            return torch.empty((0, 0), dtype=torch.float32)
        features_tensor = torch.stack(features)
        return features_tensor

    def construct_new_features(self, query):
        query_embedding = torch.tensor(np.array(get_sentence_embedding(query)), dtype=torch.float32)
        # query_features: List[torch.Tensor] = []

        # for node_id in self.nodes:
        #     if node_id == self.virtual_task_id:
        #         encoded = self.node_encoder(query_embedding)
        #         query_features.append(encoded)
        #     else:
        #         query_features.append(query_embedding.clone())

        # query_feature_tensor = torch.stack(query_features)
        # new_features = query_feature_tensor
        new_features = torch.cat((self.features, query_embedding.reshape(1, -1)), dim=0)
        # new_features = torch.cat((self.features, query_feature_tensor), dim=1)
        self.tilde_X = new_features
        return new_features

    # def construct_anchor_adj_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
    #     anchor_edges: List[Tuple[str, str]] = self.prompt_set.get_anchor_topology()
    #     num_nodes = self.num_nodes
    #     anchor_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    #     role_2_id: Dict[str, List[int]] = {}

    #     for i, node_id in enumerate(self.nodes):
    #         role = self.nodes[node_id].role
    #         role_2_id.setdefault(role, []).append(i)

    #     for edge in anchor_edges:
    #         in_role, out_role = edge
    #         in_ids = role_2_id.get(in_role, [])
    #         out_ids = role_2_id.get(out_role, [])
    #         for in_id in in_ids:
    #             for out_id in out_ids:
    #                 anchor_adj[in_id][out_id] = 1.0

    #     if self.virtual_task_id is not None:
    #         if self.virtual_task_index is not None:
    #             task_index = self.virtual_task_index
    #         else:
    #             task_index = list(self.nodes.keys()).index(self.virtual_task_id)
    #         for idx, node_id in enumerate(self.nodes):
    #             if node_id == self.virtual_task_id:
    #                 continue
    #             anchor_adj[task_index][idx] = 1.0
    #             anchor_adj[idx][task_index] = 1.0

    #     edge_index, _ = dense_to_sparse(anchor_adj)
    #     return anchor_adj, edge_index
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                added_node = self.add_node(agent_instance)
                # self.agent_node_ids.append(added_node.id)

        # if self.task_node is not None:
        #     added_task = self.add_node(self.task_node)
        #     self.virtual_task_id = added_task.id
        #     self.virtual_task_index = len(self.nodes) - 1
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))

    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        if isinstance(inputs, dict) and 'task' in inputs:
            self.prepare_probabilities(inputs['task'])

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        self.prepare_probabilities(input['task'])

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {str(e)}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs

    def encode(self, query: str) -> Tuple[torch.Tensor, torch.Tensor]:
        new_features = self.construct_new_features(query)
        latent_features = self.gcn(new_features, self.role_adj_matrix)
        latent_features = self.mlp(latent_features)
        mu = self.encoder_mu(latent_features)
        logvar = self.encoder_logvar(latent_features)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _offdiag_indices(n: int, device, dtype=torch.bool):
        eye = torch.eye(n, dtype=dtype, device=device)
        return torch.where(~eye)

    def ps(self, latent: torch.Tensor, tau: float) -> torch.Tensor:
        device, dtype = latent.device, latent.dtype
        n_plus_1, h = latent.shape
        n = n_plus_1 - 1

        h_task = latent[-1]                               # (h,)
        nodes  = latent[:-1]                              # (n, h)

        idx_i, idx_j = self._offdiag_indices(n, device)
        hi = nodes[idx_i]                                 # (m, h)
        hj = nodes[idx_j]                                 # (m, h)
        ht = h_task.expand(hi.size(0), h)                 # (m, h)
        x  = torch.cat([hi, hj, ht], dim=-1)              # (m, 3h)

        w = self.ps_linear(x).squeeze(-1)

        g = torch.logit(torch.rand_like(w), eps=1e-6)     # (m,)

        logits_off = (w + g) / tau                       # (m,)
        logits_off = logits_off.clamp(min=-12, max=12)
        if not torch.isfinite(logits_off).all():
            print(f"[NaNGuard] logits_off has NaN/Inf")
        if not torch.isfinite(w).all():
            print(f"[NaNGuard] w has NaN/Inf")
        if not torch.isfinite(g).all():
            print(f"[NaNGuard] g has NaN/Inf")

        A = torch.full((n, n), fill_value=-30.0, device=device, dtype=dtype)
        A[idx_i, idx_j] = logits_off

        P = torch.sigmoid(A)                              # (n, n)
        if not torch.isfinite(P).all():
            print(f"[NaNGuard] P has NaN/Inf")
        return P
        # transformed = self.ps_linear(latent)
        # similarity = torch.matmul(transformed, transformed.transpose(0, 1))
        # if self.ps_linear.training:
        #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity) + 1e-9) + 1e-9)
        #     logits = (similarity + gumbel_noise) / tau
        #     return torch.sigmoid(logits)
        # return torch.sigmoid(similarity)

    def qc(self, sketch_adj: torch.Tensor) -> torch.Tensor:
        refined, losses = self.refine(sketch_adj, self.role_adj_matrix)
        # store the most recent refine losses for downstream objectives
        self.refine_losses = {key: value for key, value in losses.items() if key in {"L_anchor", "L_sparse", "L_total"}}
        return refined

    def prepare_probabilities(self, query: str) -> None:
        self.mu, self.logvar = self.encode(query)
        self.latent_z = self.reparameterize(self.mu, self.logvar)
        sketch_adj = self.ps(self.latent_z, self.gumbel_tau)
        self.tilde_S = self.qc(sketch_adj)
        flat_probs = self.tilde_S.reshape(-1)
        if self.optimized_spatial:
            self.spatial_logits = torch.logit(flat_probs, eps=1e-6)
        # if self.optimized_temporal:
        #     self.temporal_edge_probs = flat_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
