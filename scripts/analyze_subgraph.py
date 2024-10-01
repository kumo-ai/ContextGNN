import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything

from hybridgnn.utils import GloveTextEmbedding
from torch_frame.testing.text_embedder import HashTextEmbedder

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-amazon")
parser.add_argument("--task", type=str, default="user-item-rate")
parser.add_argument(
    "--model",
    type=str,
    default="hybridgnn",
    choices=["hybridgnn", "idgnn", "shallowrhsgnn"],
)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--split", type=str, default="test")
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
    int(args.num_neighbors // 2**i) for i in range(args.num_layers)
]

train_table = task.get_table("train")
val_table = task.get_table(args.split)
train_df = train_table.df.groupby(task.src_entity_col, as_index=False).agg(
    {task.dst_entity_col: lambda x: set().union(*map(set, x))})
df = train_df


src = task.src_entity_col
dst = task.dst_entity_col
# Step 1: Explode the dst column
df_exploded = df.explode(dst)

# Step 2: Create reverse mapping from sponsor_id to facility_id
dst_to_src = df_exploded.groupby(dst)[src].apply(set).reset_index()

# Make sure everything is a set
dst_to_src[src] = dst_to_src[src].apply(lambda x: set([x]) if not isinstance(x, set) else x)


# Step 3: Create a dst to src dictionary
dst_to_src_dict = dst_to_src.set_index(dst)[src].to_dict()

# Step 4: First hop - src -> dst -> src
df_exploded['connected_src'] = df_exploded[dst].map(dst_to_src_dict)

df_exploded['connected_src'] = df_exploded['connected_src'].apply(lambda x: set(x) if not isinstance(x, set) else x)

# Step 5: Aggregate to find all unique connected src per original src
df_aggregated = df_exploded.groupby(src)['connected_src'].apply(lambda x: set().union(*x)).reset_index()


# Step 6: Find all dst for these connected src src -> dst -> src -> dst
df_aggregated['connected_dst'] = df_aggregated['connected_src'].apply(
    lambda x: set(df_exploded[df_exploded[src].isin(x)][dst])
)

# Result
df_result = df_aggregated[[dst, 'connected_dst']]
df_result.to_csv(f"{args.dataset}_{args.task}_second_hop.csv")

# Check the val set, how many are in right 1-hop neighbor and how many are in 2-hop neighbor
val_df = val_table.df
