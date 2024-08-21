import pytest
import torch_frame
from relbench.base.task_base import TaskType
from relbench.datasets.fake import FakeDataset
from relbench.modeling.graph import (
    get_link_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks.amazon import UserItemPurchaseTask
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder

from hybridgnn.nn.rhs_embedding import RHSEmbedding
from hybridgnn.utils import RHSEmbeddingMode


@pytest.mark.parametrize('emb_mode', list(RHSEmbeddingMode))
def test_rhs_embedding(tmp_path, emb_mode):
    dataset = FakeDataset()

    db = dataset.get_db()
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        get_stype_proposal(db),
        text_embedder_cfg=TextEmbedderConfig(text_embedder=HashTextEmbedder(8),
                                             batch_size=None),
        cache_dir=tmp_path,
    )
    task = UserItemPurchaseTask(dataset)
    assert task.task_type == TaskType.LINK_PREDICTION
    train_table = task.get_table("train")
    train_table_input = get_link_train_table_input(train_table, task)
    embedding_dim = 8
    feat = data[task.dst_entity_table]['tf']
    model = RHSEmbedding(
        emb_mode=emb_mode, num_nodes=train_table_input.num_dst_nodes,
        embedding_dim=embedding_dim, col_stats=col_stats_dict['product'],
        col_names_dict=data['product']['tf'].col_names_dict,
        stype_encoder_dict={
            torch_frame.categorical:
            torch_frame.nn.EmbeddingEncoder(),
            torch_frame.numerical:
            torch_frame.nn.LinearEncoder(),
            torch_frame.multicategorical:
            torch_frame.nn.MultiCategoricalEmbeddingEncoder(),
            torch_frame.embedding:
            torch_frame.nn.LinearEmbeddingEncoder(),
        }, feat=feat)

    out = model()
    assert out.shape[0] == train_table_input.num_dst_nodes
    assert out.shape[1] == embedding_dim
