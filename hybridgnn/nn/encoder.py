from typing import Any, Dict, List, Optional

import torch
import torch_frame
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
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
SECONDS_IN_A_DAY = 60 * 60 * 24
SECONDS_IN_A_WEEK = 7 * 60 * 60 * 24
SECONDS_IN_A_HOUR = 60 * 60
SECONDS_IN_A_MINUTE = 60


class HeteroEncoder(torch.nn.Module):
    r"""HeteroStypeWiseEncoder is a simple encoder to encode multi-modal
    data from different node types.

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
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
    """
    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype,
                                                    List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Optional[Dict[str, Any]] = None,
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

            self.encoders[node_type] = torch_frame_model_cls(
                **torch_frame_model_kwargs,
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
            node_type: self.encoders[node_type](tf)
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

        time_dim = 3 # hour, day, week
        self.time_fuser = torch.nn.Linear(time_dim, channels)

    def reset_parameters(self) -> None:
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        self.time_fuser.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time

            # rel_day = rel_time / SECONDS_IN_A_DAY
            # x = self.encoder_dict[node_type](rel_day)
            # x = self.encoder_dict[node_type](rel_hour)
            rel_hour = (rel_time // SECONDS_IN_A_HOUR).view(-1,1)
            rel_day = (rel_time // SECONDS_IN_A_DAY).view(-1,1)
            rel_week = (rel_time // SECONDS_IN_A_WEEK).view(-1,1)
            time_embed = torch.cat((rel_hour, rel_day, rel_week),dim=1).float()

            #! might need to normalize hour, day, week into the same scale
            time_embed = torch.nn.functional.normalize(time_embed, p=2.0, dim=1)
            x = self.time_fuser(time_embed)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict
