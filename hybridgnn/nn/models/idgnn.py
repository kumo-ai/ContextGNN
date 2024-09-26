from typing import Any, Dict, Optional, Type

import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from hybridgnn.nn.encoder import (
    DEFAULT_STYPE_ENCODER_DICT,
    HeteroEncoder,
    HeteroTemporalEncoder,
)
from hybridgnn.nn.models import HeteroGraphSAGE


class IDGNN(torch.nn.Module):
    r"""Implementation of IDGNN model."""
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str = 'sum',
        norm: str = 'layer_norm',
        torch_frame_model_cls: Type[torch.nn.Module] = ResNet,
        torch_frame_model_kwargs: Optional[Dict[str, Any]] = None,
        is_static=True,
    ) -> None:
        super().__init__()

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
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.is_static = is_static
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.id_awareness_emb.reset_parameters()

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
        if self.is_static is not True:
            rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                                  batch.batch_dict)

            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])
