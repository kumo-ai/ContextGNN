import pandas as pd
import pytest
import torch
import torch_frame
from torch_frame.data import Dataset

from hybridgnn.nn.rhs_embedding import RHSEmbedding
from hybridgnn.utils import RHSEmbeddingMode


@pytest.mark.parametrize('emb_mode', list(RHSEmbeddingMode))
def test_rhs_embedding(emb_mode):
    df = pd.DataFrame({'A': [0.1, 0.2, 0.3, 0.4, 0.5], 'B': [0, 1, 0, 1, 0]})

    col_to_stype = {'A': torch_frame.numerical, 'B': torch_frame.categorical}
    dataset = Dataset(df, col_to_stype).materialize()
    model = RHSEmbedding(
        emb_mode=emb_mode, num_nodes=5, embedding_dim=8,
        col_stats=dataset.col_stats,
        col_names_dict=dataset.tensor_frame.col_names_dict,
        stype_encoder_dict={
            torch_frame.categorical: torch_frame.nn.EmbeddingEncoder(),
            torch_frame.numerical: torch_frame.nn.LinearEncoder(),
        })
    index = torch.tensor([1, 3, 2, 4, 0])

    out = model(index, dataset.tensor_frame)
    assert out.shape[0] == len(index)
    assert out.shape[1] == 8
