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
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
    
    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.rhs.reset_parameters()
        self.lhs.reset_parameters()
        self.w0.reset_parameters()

    def forward(
        self,
        src_tensor: Tensor,
        dst_tensor: Tensor,
    ) -> Tensor:
        lhs_embedding = self.lhs()
        rhs_embedding = self.rhs()
        mask = torch.zeros(self.num_src_nodes, self.num_dst_nodes).to(src_tensor.device)
        mask[src_tensor][dst_tensor] = 1
        mat = lhs_embedding @ rhs_embedding.t()
        return ((1 - mat[mask]) **2).sum() + self.w0*(mat[~mask]**2).sum()