from typing_extensions import Self
from hybridgnn.nn.encoder import DEFAULT_STYPE_ENCODER_DICT

from hybridgnn.nn.rhs_embedding import RHSEmbedding
from typing import Any, Dict

import torch
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData

from hybridgnn.nn.rhs_embedding import RHSEmbedding
from hybridgnn.utils import RHSEmbeddingMode
class RHSEmbeddingGNN(torch.nn.Module):
    def __init__(self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        rhs_emb_mode: RHSEmbeddingMode,
        dst_entity_table: str,
        num_nodes: int,
        embedding_dim: int,
    ) :
        stype_encoder_dict = {
            k: v[0]()
            for k, v in DEFAULT_STYPE_ENCODER_DICT.items()
            if k in data[dst_entity_table]['tf'].col_names_dict.keys()
        }
        self.rhs_embedding = RHSEmbedding(
            emb_mode=rhs_emb_mode,
            embedding_dim=embedding_dim,
            num_nodes=num_nodes,
            col_stats=col_stats_dict[dst_entity_table],
            col_names_dict=data[dst_entity_table]['tf'].col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            feat=data[dst_entity_table]['tf'],
        )

    def reset_parameters(self):
        self.rhs_embedding.reset_parameters()

    def to(self, *args, **kwargs) -> Self:
        # Explicitly call `to` on the RHS embedding to move caches to the
        # device. 
        self.rhs_embedding.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    

    def cpu(self) -> Self:
        self.rhs_embedding.cpu()
        return super().cpu()
    
    def cuda(self, *args, **kwargs) -> Self:
        self.rhs_embedding.cuda(*args, **kwargs)
        return super().cuda(*args, **kwargs)

    

