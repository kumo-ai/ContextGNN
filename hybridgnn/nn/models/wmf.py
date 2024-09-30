import torch
from typing import Any, Dict, Optional, Type

import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

class WeightedMatrixFactorization(torch.nn.Module):
    def __init__(
        self,
        num_src_nodes: int,
        num_dst_nodes: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.rhs = torch.nn.Embedding(num_src_nodes, embedding_dim)
        self.lhs = torch.nn.Embedding(num_dst_nodes, embedding_dim)
        self.w0 = torch.nn.Parameter(torch.tensor(1.0))
    
    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.rhs.reset_parameters()
        self.lhs.reset_parameters()
        self.w0.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        ground_truth: Tensor,
    ) -> Dict[NodeType, Tensor]:
        batch_size = batch[entity_table].seed_time.size(0)
        lhs_idx = batch[entity_table].n_id[:batch_size]
        lhs_embedding = self.lhs(lhs_idx)
        mat = lhs_embedding @ self.rhs.t() 
        return torch.sum((1 - mat[ground_truth]) **2) + self.w0(mat[~ground_truth]**2)