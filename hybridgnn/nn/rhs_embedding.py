from typing import Literal

import kumoml
import pandas as pd
import torch
import torch_frame
from kumoapi.model_plan import RHSEmbeddingMode
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.nn.models.resnet import FCResidualBlock

from io_utils import logging
from kumo.config import Encoder
from kumo.nn import TensorFrameEncoder

logger = logging.getLogger(__name__)


class RHSEmbedding(torch.nn.Module):
    def __init__(self,
                 emb_mode: str,
                 embedding_dim: int,
                 num_nodes: int,
                 init_std: float = 1.0,
                 encoder_dict: dict[str, Encoder] | None = None,):
        self.emb_mode = emb_mode
        self.encoder = TensorFrameEncoder(
                encoder_dict,
                is_temporal=False,
                out_channels=embedding_dim,
                act=None,
            )

        seqs: list[torch.nn.Module] = []
        if self.emb_mode == 'feature':
                seqs += [  # Apply deep RHS projector for pure feature mode.
                    FCResidualBlock(embedding_dim, embedding_dim),
                    FCResidualBlock(embedding_dim, embedding_dim),
                ]
        seqs += [torch.nn.LayerNorm(embedding_dim, eps=1e-7)]
        self.projector = torch.nn.Sequential(*seqs)
        self.lookup_embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.init_std = init_std
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
            for param in self.encoder.parameters():
                param.data *= self.init_std
        if self.projector is not None:
            for child in self.projector.children():
                child.reset_parameters()
                if isinstance(child, torch.nn.LayerNorm):
                    child.weight.data *= self.init_std

    def forward(self, index: Tensor | None = None, feat: Tensor | None = None) -> Tensor:
        outs = []
        if self.emb_mode in ["lookup", "fusion"]:
            assert index is not None
            outs.append(self.lookup_embedding(index))
        if self.emb_mode in ["feature", "fusion"]:
            out = self.encoder(feat)
            out = self.projector(out)
            outs.append(out)
        import pdb
        pdb.set_trace()
        return