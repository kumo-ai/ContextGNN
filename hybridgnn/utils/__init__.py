from .text_embedder import GloveTextEmbedding

from enum import Enum


class RHSEmbeddingMode(Enum):
    r"""Specifies how to incorporate shallow RHS representations in link
    prediction tasks.
    """
    # Use trainable look-up embeddings (transductive):
    LOOKUP = 'lookup'
    # Purely rely on shallow RHS input features (inductive):
    FEATURE = 'feature'
    # Fuse look-up embeddings and shallow RHS input features (transductive):
    FUSION = 'fusion'


__all__ = classes = ['GloveTextEmbedding', 'RHSEmbeddingMode']
