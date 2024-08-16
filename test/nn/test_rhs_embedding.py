import pandas as pd
import pytest
import torch
import torch_frame
from torch_frame.data import Dataset

from hybridgnn.nn.encoder import DEFAULT_STYPE_ENCODER_DICT
from hybridgnn.nn.rhs_embedding import RHSEmbedding


@pytest.mark.parametrize('emb_mode', ['fusion', 'lookup', 'feature'])
def test_rhs_embedding(emb_mode):
    df = pd.DataFrame({'A': [0.1, 0.2, 0.3, 0.4, 0.5], 'B': [0, 1, 0, 1, 0]})

    col_to_stype = {'A': torch_frame.numerical, 'B': torch_frame.categorical}
    dataset = Dataset(df, col_to_stype).materialize()

    encoder_dict = DEFAULT_STYPE_ENCODER_DICT

    model = RHSEmbedding(
        emb_mode=emb_mode,
        num_nodes=5,
        embedding_dim=8,
        encoder_dict=encoder_dict,
    )
    index = torch.tensor([1, 3])

    model(index, dataset.tf)
