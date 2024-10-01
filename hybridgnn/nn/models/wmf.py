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
        self.rhs = torch.nn.Embedding(num_dst_nodes, embedding_dim)
        self.lhs = torch.nn.Embedding(num_src_nodes, embedding_dim)
        self.w0 = torch.nn.Parameter(torch.tensor(1.0))
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.register_buffer("full_lhs", torch.arange(0, self.num_src_nodes))
        self.register_buffer("full_rhs", torch.arange(0, self.num_dst_nodes))
    
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
        lhs_embedding = self.lhs(src_tensor)
        rhs_embedding = self.rhs(dst_tensor)
        mat_pos = lhs_embedding @ rhs_embedding.t()

        mask = ~torch.isin(self.full_lhs, src_tensor)

        # Filter out the values present in the first tensor
        neg_lhs = self.full_lhs[mask]
        mask = ~torch.isin(self.full_rhs, dst_tensor)
        neg_rhs = self.full_rhs[mask]
        mat_neg = torch.mm(self.lhs(neg_lhs), self.rhs(neg_rhs).t())
        return ((1.0 - mat_pos) **2).sum() + self.w0*((mat_neg**2).sum())