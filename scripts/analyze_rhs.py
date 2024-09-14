import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_geometric.seed import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-trial")
parser.add_argument("--task", type=str, default="site-sponsor-run")
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

df_exploded = train_table.df.explode('sponsor_id')

product_counts = df_exploded['sponsor_id'].value_counts()

# Step 3: Filter product_ids that appear only once
product_once = product_counts[product_counts == 1]

# Output the number of product_ids that appear only once
num_products_once = len(product_once)
print(f'Number of sponsor_ids that appear only once in training'
      f' table: {num_products_once}')

total_unique_products = df_exploded['sponsor_id'].nunique()

print(f'Total number of unique sponsor_ids: {total_unique_products}')

all_items = [
    item for sublist in train_table.df['sponsor_id'] for item in sublist
]

# Assuming you have the 'all_items' from your data
# Count the frequency of each item
item_counts = Counter(all_items)

# Get the list of frequencies
frequencies = list(item_counts.values())

# Calculate quantiles (e.g., 25th, 50th, 75th percentiles)
quantiles = np.quantile(frequencies, [0.25, 0.5, 0.75])
print(f"25th Percentile (Q1): {quantiles[0]}")
print(f"50th Percentile (Median): {quantiles[1]}")
print(f"75th Percentile (Q3): {quantiles[2]}")

# Plot the histogram of item frequencies
plt.figure(figsize=(8, 6))
plt.hist(frequencies, bins=range(1, max(frequencies) + 2), edgecolor='black')
plt.title('Distribution of Items Being Rated Across All Users')
plt.xlabel('Number of Times Item Was Rated')
plt.ylabel('Frequency of Items')
plt.grid(True)

# Save the plot to a file (optional)
plt.savefig(f'{args.dataset}_{args.task}_dst_node_distribution.png')

# Show the plot
plt.show()
