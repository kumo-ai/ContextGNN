import torch
from relbench.datasets.fake import FakeDataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder

from contextgnn.nn.encoder import DEFAULT_STYPE_ENCODER_DICT, HeteroEncoder


def test_encoder(tmp_path):
    dataset = FakeDataset()

    db = dataset.get_db()
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        get_stype_proposal(db),
        text_embedder_cfg=TextEmbedderConfig(text_embedder=HashTextEmbedder(8),
                                             batch_size=None),
        cache_dir=tmp_path,
    )
    node_to_col_names_dict = {
        node_type: data[node_type].tf.col_names_dict
        for node_type in data.node_types
    }

    # Ensure that full-batch model works as expected ##########################

    encoder = HeteroEncoder(
        64,
        node_to_col_names_dict,
        col_stats_dict,
        stype_encoder_cls_kwargs=DEFAULT_STYPE_ENCODER_DICT,
        torch_frame_model_kwargs={
            "channels": 128,
            "num_layers": 4,
        },
    )

    x_dict = encoder(data.tf_dict)
    assert 'product' in x_dict.keys()
    assert 'customer' in x_dict.keys()
    assert 'review' in x_dict.keys()
    assert 'relations' in x_dict.keys()
    assert x_dict['relations'].shape == torch.Size([20, 64])
    assert x_dict['product'].shape == torch.Size([30, 64])
