from typing import Any, Dict, Optional, Type

import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from typing_extensions import Self

from hybridgnn.nn.encoder import (
    DEFAULT_STYPE_ENCODER_DICT,
    HeteroEncoder,
    HeteroTemporalEncoder,
)
from hybridgnn.nn.models import HeteroGraphSAGE
from hybridgnn.nn.models.rhsembeddinggnn import RHSEmbeddingGNN
from hybridgnn.utils import RHSEmbeddingMode


class ShallowRHSGNN(RHSEmbeddingGNN):
    r"""Implementation of ShallowRHSGNN model."""
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        rhs_emb_mode: RHSEmbeddingMode,
        dst_entity_table: str,
        num_nodes: int,
        num_layers: int,
        channels: int,
        embedding_dim: int,
        aggr: str = 'sum',
        norm: str = 'layer_norm',
        torch_frame_model_cls: Type[torch.nn.Module] = ResNet,
        torch_frame_model_kwargs: Optional[Dict[str, Any]] = None,
        is_static: Optional[bool] = False,
    ) -> None:
        super().__init__(data, col_stats_dict, rhs_emb_mode, dst_entity_table,
                         num_nodes, embedding_dim)
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            stype_encoder_cls_kwargs=DEFAULT_STYPE_ENCODER_DICT,
            torch_frame_model_cls=torch_frame_model_cls,
            torch_frame_model_kwargs=torch_frame_model_kwargs,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=1,
            norm=norm,
            num_layers=1,
        )
        self.lhs_projector = torch.nn.Linear(channels, embedding_dim)
        self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.is_static = is_static

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.id_awareness_emb.reset_parameters()
        self.lhs_projector.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        # Add ID-awareness to the root node
        x_dict[entity_table][:seed_time.size(0
                                             )] += self.id_awareness_emb.weight

        if not self.is_static:
            rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                                  batch.batch_dict)

            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        batch_size = seed_time.size(0)
        lhs_emb = self.lhs_projector(x_dict[entity_table][:batch_size])
        rhs_emb = self.rhs_embedding()
        return lhs_emb @ rhs_emb.t()

    def to(self, *args, **kwargs) -> Self:
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        return super().cuda(*args, **kwargs)
