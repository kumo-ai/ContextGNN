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
from torch_geometric.loader import NeighborLoader

from hybridgnn.nn.models import IDGNN, HybridGNN


def test_idgnn(tmp_path):
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
    batch_size = 16
    train_loader = NeighborLoader(
        data,
        num_neighbors=[128, 128],
        time_attr="time",
        input_nodes=train_table_input.src_nodes,
        input_time=train_table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=batch_size,
        temporal_strategy='last',
        shuffle=True,
    )

    batch = next(iter(train_loader))

    assert len(batch[task.dst_entity_table].batch) > 0
    model = IDGNN(data=data, col_stats_dict=col_stats_dict, num_layers=2,
                  channels=64, out_channels=1, aggr="sum", norm="layer_norm")
    model.train()

    out = model(batch, task.src_entity_table, task.dst_entity_table).flatten()
    assert len(out) == len(batch[task.dst_entity_table].n_id)


def test_hybridgnn(tmp_path):
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
    batch_size = 16
    train_loader = NeighborLoader(
        data,
        num_neighbors=[128, 128],
        time_attr="time",
        input_nodes=train_table_input.src_nodes,
        input_time=train_table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=batch_size,
        temporal_strategy='last',
        shuffle=True,
    )

    batch = next(iter(train_loader))

    assert len(batch[task.dst_entity_table].batch) > 0

    channels = 16
    embedding_dim = 8
    model = HybridGNN(data=data, col_stats_dict=col_stats_dict,
                      num_nodes=train_table_input.num_dst_nodes, num_layers=2,
                      channels=channels, aggr="sum", norm="layer_norm",
                      embedding_dim=embedding_dim)
    model.train()

    logits = model(batch, task.src_entity_table, task.dst_entity_table)

    assert logits.shape[0] == batch_size
    assert logits.shape[1] == train_table_input.num_dst_nodes
