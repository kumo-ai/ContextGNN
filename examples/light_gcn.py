"""Example script to run the models in this repository.

python3 light_gcn.py --dataset rel-hm --task user-item-purchase --val_loss
python3 light_gcn.py --dataset rel-avito --task user-ad-visit --val_loss
"""

from __future__ import annotations

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
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.nn.models import LightGCN
from torch_geometric.seed import seed_everything
from torch_geometric.utils import coalesce, to_undirected
from tqdm import tqdm

from contextgnn.utils import GloveTextEmbedding

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-trial")
parser.add_argument("--task", type=str, default="site-sponsor-run")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_num_train_edges", type=int, default=3000000)
parser.add_argument("--lambda_reg", type=float, default=1e-4)
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--val_loss", default=False, action="store_true")
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
split_edge_weight_dict: Dict[str, Tensor] = {}
n_id_dict: Dict[str, Tensor] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_link_train_table_input(table, task)

    # Get n_id for each split #################################################
    src_entities = torch.from_numpy(table.df[task.src_entity_col].to_numpy())
    # Only validation and test need the source entities for prediction
    if split != "train":
        n_id_dict[split] = src_entities

    # Get message passing edge_index for each split ###########################
    dst_csr = table_input.dst_nodes[1]
    # Compute counts per row from the CSR matrix
    counts_per_row = dst_csr.crow_indices()[1:] - dst_csr.crow_indices()[:-1]
    # Get source nodes using row indices
    src = table_input.src_nodes[1].repeat_interleave(counts_per_row)
    # Get edge_index using src and column indices
    edge_index = torch.stack([src, dst_csr.col_indices()], dim=0)
    # Convert to bipartite graph
    edge_index[1, :] += num_src_nodes
    # Remove duplicated edges but use edge weight for message passing
    edge_weight = torch.ones(edge_index.size(1)).to(edge_index.device)
    edge_index, edge_weight = coalesce(edge_index, edge_attr=edge_weight,
                                       num_nodes=num_total_nodes)
    split_edge_index_dict[split] = edge_index
    split_edge_weight_dict[split] = edge_weight

model = LightGCN(num_total_nodes, embedding_dim=args.channels,
                 num_layers=args.num_layers).to(device)

train_edge_index = split_edge_index_dict["train"].to("cpu")
train_edge_weight = split_edge_weight_dict["train"].to("cpu")
# Shuffle train edges to avoid only using same edges for supervision each time
perm = torch.randperm(train_edge_index.size(1), device="cpu")
train_edge_index = train_edge_index[:, perm][:, :args.max_num_train_edges].to(
    device)
train_edge_weight = train_edge_weight[perm][:args.max_num_train_edges].to(
    device)
# Convert to undirected graph
train_mp_edge_index, train_mp_edge_weight = to_undirected(
    train_edge_index, train_edge_weight)
train_mp_edge_index = train_mp_edge_index.to(device)
train_mp_edge_weight = train_mp_edge_weight.to(device)
val_edge_index = split_edge_index_dict["val"].to(device)
val_edge_weight = split_edge_weight_dict["val"].to(device)
val_mp_edge_index, val_mp_edge_weight = to_undirected(val_edge_index,
                                                      val_edge_weight)
val_mp_edge_index = val_mp_edge_index.to(device)
val_mp_edge_weight = val_mp_edge_weight.to(device)

val_n_ids = n_id_dict["val"].to(device)
test_n_ids = n_id_dict["test"].to(device)
train_loader: DataLoader = DataLoader(
    torch.arange(train_edge_index.size(1)),  # type: ignore
    shuffle=True,
    batch_size=args.batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
writer = SummaryWriter()


def get_edge_label_index(sup_edge_index: Tensor, index: Tensor) -> Tensor:
    pos_edge_label_index = sup_edge_index[:, index].to(device)
    neg_edge_label_index = torch.stack([
        pos_edge_label_index[0],
        torch.randint(
            num_src_nodes,
            num_total_nodes,
            (index.numel(), ),
            device=device,
        )
    ], dim=0)
    edge_label_index = torch.cat([
        pos_edge_label_index,
        neg_edge_label_index,
    ], dim=1)
    return edge_label_index


def train(epoch: int) -> float:
    model.train()
    total_loss = total_examples = 0
    total_steps = min(args.max_steps_per_epoch, len(train_loader))
    for i, index in enumerate(
            tqdm(train_loader, total=total_steps, desc="Train")):
        if i >= args.max_steps_per_epoch:
            break
        edge_label_index = get_edge_label_index(train_edge_index, index)
        optimizer.zero_grad()
        pos_rank, neg_rank = model(
            train_mp_edge_index,
            edge_label_index,
            edge_weight=train_mp_edge_weight,
        ).chunk(2)
        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
            lambda_reg=args.lambda_reg,
        )
        loss.backward()
        writer.add_scalar("Step loss/train", float(loss),
                          total_steps * (epoch - 1) + i)
        writer.flush()
        optimizer.step()
        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(
    stage: Literal["val", "test"],
    n_ids: Tensor,
    desc: str,
    epoch: None | int = None,
) -> np.ndarray:
    model.eval()
    mp_edge_index = train_mp_edge_index
    mp_edge_weight = train_mp_edge_weight
    if stage == "test":
        # For test set we use both train and val edges for message passing
        mp_edge_index = torch.cat([mp_edge_index, val_mp_edge_index], dim=1)
        mp_edge_weight = torch.cat(
            [mp_edge_weight,
             val_mp_edge_weight.to(mp_edge_index.device)])
        # Remove duplicated edges used for message passing
        mp_edge_index, mp_edge_weight = coalesce(
            mp_edge_index,
            edge_attr=mp_edge_weight,
            num_nodes=num_total_nodes,
        )
    elif args.val_loss and stage == "val":
        total_loss = total_examples = 0
        for start in tqdm(range(0, val_edge_index.size(1), args.batch_size),
                          desc=desc):
            end = min(start + args.batch_size, val_edge_index.size(1))
            index = torch.arange(start, end)
            edge_label_index = get_edge_label_index(val_edge_index, index)
            pos_rank, neg_rank = model(
                mp_edge_index,
                edge_label_index,
                edge_weight=mp_edge_weight,
            ).chunk(2)
            loss = model.recommendation_loss(
                pos_rank,
                neg_rank,
                node_id=edge_label_index.unique(),
                lambda_reg=args.lambda_reg,
            )
            total_loss += float(loss) * pos_rank.numel()
            total_examples += pos_rank.numel()
        val_loss = total_loss / total_examples
        if epoch is not None:
            writer.add_scalar("Total loss/val", float(val_loss), epoch)
            writer.flush()
    emb = model.get_embedding(mp_edge_index, edge_weight=mp_edge_weight)
    src_emb, dst_emb = emb[:num_src_nodes], emb[num_src_nodes:]
    pred_list: List[Tensor] = []
    # Make prediction in minibatch
    for start in tqdm(range(0, n_ids.size(0), args.batch_size), desc=desc):
        end = start + args.batch_size
        logits = src_emb[n_ids[start:end]] @ dst_emb.t()
        _, pred_mini = torch.topk(logits, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


state_dict = None
best_val_metric = 0

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    writer.add_scalar("Total loss/train", float(train_loss), epoch)
    writer.flush()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(
            stage="val",
            n_ids=val_n_ids,
            desc="Val",
            epoch=epoch,
        )
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
              f"Val metrics: {val_metrics}")
        for metric in val_metrics:
            writer.add_scalar(
                f"{metric}/val",
                float(val_metrics[metric]),
                epoch,
            )
            writer.flush()
        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

assert state_dict is not None
model.load_state_dict(state_dict)
val_pred = test(
    stage="val",
    n_ids=val_n_ids,
    desc="Val",
)
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best val metrics: {val_metrics}")

test_pred = test(
    stage="test",
    n_ids=test_n_ids.to(device),
    desc="Test",
)
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
writer.close()
