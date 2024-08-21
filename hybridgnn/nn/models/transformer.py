import torch
import math
from torch import Tensor, nn
from torch_geometric.typing import EdgeType, NodeType
from torch.nested import nested_tensor

from torch_geometric.nn.aggr.utils import MultiheadAttentionBlock
from torch_geometric.utils import to_dense_batch, to_nested_tensor, from_nested_tensor
from torch_geometric.utils import cumsum, scatter
from torch_geometric.nn.encoding import PositionalEncoding


class Transformer(torch.nn.Module):
    r"""A module to attend to rhs embeddings with a transformer. 
    Args:
        in_channels (int): The number of input channels of the RHS embedding.
        out_channels (int): The number of output channels. 
        hidden_channels (int): The hidden channel dimension of the transformer. 
        heads (int): The number of attention heads for the transformer.
        num_transformer_blocks (int): The number of transformer blocks. 
        dropout (float): dropout rate for the transformer 
        position_encoding (str): type of positional encoding,
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        heads: int = 1,
        num_transformer_blocks: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.lin = torch.nn.Linear(in_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.blocks = torch.nn.ModuleList([
            MultiheadAttentionBlock(
                channels=hidden_channels,
                heads=heads,
                layer_norm=True,
                dropout=dropout,
            ) for _ in range(num_transformer_blocks)
        ])

    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()
        self.lin.reset_parameters()
        self.fc.reset_parameters()


    def forward(self, rhs_embed: Tensor, index: Tensor, batch_size) -> Tensor:
        r"""Returns the attended to rhs embeddings
        """
        rhs_embed = self.lin(rhs_embed)

        # #! if we sort the index, we need to sort the rhs_embed
        sorted_index, sorted_idx = torch.sort(index, stable=True)
        index = index[sorted_idx]
        rhs_embed = rhs_embed[sorted_idx]
        reverse = self.inverse_permutation(sorted_idx)

        x, mask = to_dense_batch(rhs_embed, index, batch_size=batch_size)
        for block in self.blocks:
            x = block(x, x)
        x = x[mask]
        x = x.view(-1, self.hidden_channels)
        x = x[reverse]
        return self.fc(x)

    def inverse_permutation(self,perm):
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=perm.device)
        return inv

