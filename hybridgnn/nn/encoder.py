from typing import Any, Dict, List

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from torch_geometric.nn import PositionalEncoding
from torch_geometric.typing import NodeType

DEFAULT_STYPE_ENCODER_DICT: Dict[torch_frame.stype, Any] = {
    torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
    torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
    torch_frame.multicategorical: (
        torch_frame.nn.MultiCategoricalEmbeddingEncoder,
        {},
    ),
    torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
    torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
}


class HeteroStypeWiseEncoder(torch.nn.Module):
    r"""StypeWiseEncoder based on PyTorch Frame.

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):  # noqa: E501
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        node_to_col_stats (Dict[NodeType, Dict[str, Dict[StatType, Any]]]):
            A dictionary mapping from node type to column statistics dictionary
            compatible to PyTorch Frame.
        stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """
    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype,
                                                    List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        stype_encoder_cls_kwargs: Dict[torch_frame.stype,
                                       Any] = DEFAULT_STYPE_ENCODER_DICT,
    ) -> None:
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype:
                stype_encoder_cls_kwargs[stype][0](
                    **stype_encoder_cls_kwargs[stype][1])
                for stype in node_to_col_names_dict[node_type].keys()
            }

            self.encoders[node_type] = StypeWiseFeatureEncoder(
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )

    def reset_parameters(self) -> None:
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        x_dict = {
            node_type: self.encoders[node_type](tf)[0].sum(axis=1)
            for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroTemporalEncoder(torch.nn.Module):
    def __init__(self, node_types: List[NodeType], channels: int) -> None:
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict({
            node_type:
            PositionalEncoding(channels)
            for node_type in node_types
        })
        self.lin_dict = torch.nn.ModuleDict({
            node_type:
            torch.nn.Linear(channels, channels)
            for node_type in node_types
        })

    def reset_parameters(self) -> None:
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict
