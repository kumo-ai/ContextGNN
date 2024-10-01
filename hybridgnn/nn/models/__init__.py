from .graphsage import HeteroGraphSAGE
from .idgnn import IDGNN
from .hybridgnn import HybridGNN
from .shallowrhsgnn import ShallowRHSGNN
from .rhsembeddinggnn import RHSEmbeddingGNN
from .wmf import WeightedMatrixFactorization

__all__ = classes = ['HeteroGraphSAGE',
                     'IDGNN',
                     'HybridGNN',
                     'ShallowRHSGNN',
                     'RHSEmbeddingGNN',
                     'WeightedMatrixFactorization']
