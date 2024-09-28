import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import (
    get_link_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="user-item-purchase")

parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache_dir", type=str,
                    default=os.path.expanduser("~/.cache/relbench_examples"))
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=HashTextEmbedder(out_channels=args.channels,
                                       device=device), batch_size=256),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_neighbors = [
            -1 for i in range(args.num_layers)
            ]

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
num_dst_nodes_dict: Dict[str, int] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_link_train_table_input(table, task)
    dst_nodes_dict[split] = table_input.dst_nodes
    num_dst_nodes_dict[split] = table_input.num_dst_nodes
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=table_input.src_nodes,
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
val_table = task.get_table('val')
val_df = val_table.df
val_seen_percent = []
for batch in loader_dict["val"]:
    batch.to(device)
    # use ID-GNN module
    entities = batch[task.src_entity_table]['n_id'][:args.batch_size]
    # get ID-GNN rhs
    rhs = batch.n_id_dict[task.dst_entity_table]
    df = val_df[val_df[task.src_entity_col].isin(entities.tolist())]
    ground_truth = np.concatenate(df[task.dst_entity_col].values)
    idgnn_rhs = np.intersect1d(ground_truth, rhs.cpu())
    percent = len(idgnn_rhs)/len(ground_truth)
    val_seen_percent.append(percent)

test_table = task.get_table('test')
test_df = test_table.df
test_seen_percent = []
for batch in loader_dict["test"]:
    batch.to(device)
    # use ID-GNN module
    entities = batch[task.src_entity_table]['n_id']
    # get ID-GNN rhs
    rhs = batch.n_id_dict[task.dst_entity_table]
    df = test_df[test_df[task.src_entity_col].isin(entities.tolist())]
    ground_truth = np.concatenate(df[task.dst_entity_col].values)
    idgnn_rhs = np.intersect1d(ground_truth, rhs.cpu())
    percent = len(idgnn_rhs)/len(ground_truth)
    test_seen_percent.append(percent)


print(args.dataset, args.task, args.num_layers, sum(val_seen_percent)/len(val_seen_percent), sum(test_seen_percent)/len(test_seen_percent))
