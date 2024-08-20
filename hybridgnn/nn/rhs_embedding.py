from typing import List, Optional

import torch
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.nn import StypeWiseFeatureEncoder
from torch_frame.nn.models.resnet import FCResidualBlock

from hybridgnn.utils import RHSEmbeddingMode


class RHSEmbedding(torch.nn.Module):
    r"""RHSEmbedding module for GNNs."""
    def __init__(
        self,
        emb_mode: RHSEmbeddingMode,
        embedding_dim: int,
        num_nodes: int,
        col_stats: dict,
        col_names_dict: dict,
        stype_encoder_dict: dict,
        feat: Optional[TensorFrame] = None,
    ):
        super().__init__()
        self.emb_mode = emb_mode
        # Encodes the column features of a table into a shared embedding space.
        self.encoder = None
        self.projector = None
        if self.emb_mode in [
                RHSEmbeddingMode.FEATURE, RHSEmbeddingMode.FUSION
        ]:
            self.encoder = StypeWiseFeatureEncoder(
                out_channels=embedding_dim, col_stats=col_stats,
                col_names_dict=col_names_dict,
                stype_encoder_dict=stype_encoder_dict)

        seqs: List[torch.nn.Module] = []
        if self.emb_mode in [
                RHSEmbeddingMode.FEATURE, RHSEmbeddingMode.FUSION
        ]:
            if feat is None:
                raise ValueError(f"RHSEmbedding mode {self.emb_mode} "
                                 f"requires feat data.")
            self._feat = feat
            seqs += [
                FCResidualBlock(embedding_dim, embedding_dim),
                FCResidualBlock(embedding_dim, embedding_dim),
            ]
            seqs += [torch.nn.LayerNorm(embedding_dim, eps=1e-7)]
            self.projector = torch.nn.Sequential(*seqs)
        self.lookup_embedding = None
        if self.emb_mode in [RHSEmbeddingMode.FUSION, RHSEmbeddingMode.LOOKUP]:
            self.lookup_embedding = torch.nn.Embedding(num_nodes,
                                                       embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        if self.lookup_embedding is not None:
            self.lookup_embedding.reset_parameters()
        if self.encoder is not None:
            self.encoder.reset_parameters()
        if self.projector is not None:
            for child in self.projector.children():
                child.reset_parameters()

    def forward(self) -> Tensor:
        outs = []
        if self.lookup_embedding is not None:
            outs.append(self.lookup_embedding.weight)
        if self.encoder is not None and self.projector is not None:
            assert self._feat is not None

            out = self.encoder(self._feat)[0]
            out = self.projector(out)
            # fuse
            out = torch.sum(out, dim=1)
            outs.append(out)
        result = sum(outs)
        assert isinstance(result, Tensor)
        return result
