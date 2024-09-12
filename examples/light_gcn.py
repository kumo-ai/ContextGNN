"""Example script to run the models in this repository.

python3 light_gcn.py --dataset rel-hm --task user-item-purchase
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import torch
from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import (
    get_link_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from torch import Tensor
from torch.utils.data import DataLoader
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.nn.models import LightGCN
from torch_geometric.seed import seed_everything
from torch_geometric.utils import coalesce
from tqdm import tqdm

from hybridgnn.utils import GloveTextEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-trial")
parser.add_argument("--task", type=str, default="site-sponsor-run")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=32)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_num_train_edges", type=int, default=5000000)
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

_ = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_src_nodes = task.num_src_nodes
num_dst_nodes = task.num_dst_nodes
num_total_nodes = num_src_nodes + num_dst_nodes

split_edge_index_dict: Dict[str, Tensor] = {}
n_id_dict: Dict[str, Tensor] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_link_train_table_input(table, task)

    # Get n_id for each split #################################################
    src_entities = torch.from_numpy(table.df[task.src_entity_col].to_numpy())
    if split == "train":
        # Only dedup src entities for train set
        # For evaluation we need to align with the train table order
        src_entities = src_entities.unique()
    n_id_dict[split] = src_entities

    # Get message passing edge_index for each split ###########################
    sparse_tensor = SparseTensor(table_input.dst_nodes[1], device=device)
    src, dst = sparse_tensor[torch.arange(table_input.dst_nodes[1].size(0))]
    edge_index = torch.stack([src, dst])
    edge_index[:, 1] += num_dst_nodes
    # Remove duplicated edges used for message passing
    edge_index = coalesce(edge_index, num_nodes=num_total_nodes)
    split_edge_index_dict[split] = edge_index

model = LightGCN(num_total_nodes, embedding_dim=args.channels,
                 num_layers=args.num_layers).to(device)

train_edge_index = split_edge_index_dict["train"][:, :args.max_num_train_edges]
train_edge_index = train_edge_index.to(device)
val_edge_index = split_edge_index_dict["val"].to(device)
val_n_ids = n_id_dict["val"].to(device)
test_n_ids = n_id_dict["test"].to("cpu")
train_loader: DataLoader = DataLoader(  # type: ignore[arg-type]
    range(train_edge_index.size(1)),
    shuffle=True,
    batch_size=args.batch_size,
)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()
    total_loss = total_examples = 0
    for index in tqdm(train_loader):
        pos_edge_label_index = train_edge_index[:, index].to(device)
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(
                num_src_nodes,
                num_src_nodes + num_dst_nodes,
                (index.numel(), ),
                device=device,
            )
        ], dim=0)
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)
        optimizer.zero_grad()
        pos_rank, neg_rank = model(train_edge_index, edge_label_index).chunk(2)
        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(
    stage: Literal["val", "test"],
    train_edge_index: Tensor,
    val_edge_index: Tensor,
    n_ids: Tensor,
    desc: str,
) -> np.ndarray:
    model.eval()
    mp_edge_index = train_edge_index
    if stage == "test":
        # For test set we use both train and val edges for message passing
        mp_edge_index = torch.cat([mp_edge_index, val_edge_index], dim=1)
        # Remove duplicated edges used for message passing
        mp_edge_index = coalesce(mp_edge_index, num_nodes=num_total_nodes)
    emb = model.get_embedding(mp_edge_index)
    src_emb, dst_emb = emb[:num_src_nodes], emb[num_src_nodes:]
    pred_list: List[Tensor] = []
    for start in tqdm(range(0, n_ids.size(0), args.batch_size), desc=desc):
        end = start + args.batch_size
        n_id = n_ids[start:end]
        logits = src_emb[n_id] @ dst_emb.t()
        _, pred_mini = torch.topk(logits, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


state_dict = None
best_val_metric = 0

for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(
            stage="val",
            train_edge_index=train_edge_index,
            val_edge_index=val_edge_index,
            n_ids=val_n_ids,
            desc="Val",
        )
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
              f"Val metrics: {val_metrics}")

        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

assert state_dict is not None
model.load_state_dict(state_dict)
val_pred = test(
    stage="val",
    train_edge_index=train_edge_index,
    val_edge_index=val_edge_index,
    n_ids=val_n_ids,
    desc="Val",
)
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best val metrics: {val_metrics}")

test_pred = test(
    stage="test",
    train_edge_index=train_edge_index,
    val_edge_index=val_edge_index,
    n_ids=test_n_ids.to(device),
    desc="Test",
)
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
