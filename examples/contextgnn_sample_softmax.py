"""Example script to run sample softmax on ContextGNN.

python contextgnn_sample_softmax.py --dataset rel-trial --task site-sponsor-run
    --epochs 10
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from tqdm import tqdm

from contextgnn.nn.models import IDGNN, ContextGNN, ShallowRHSGNN
from contextgnn.utils import GloveTextEmbedding, RHSEmbeddingMode

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-amazon")
parser.add_argument("--task", type=str, default="user-item-purchase")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--num_neighbors", type=int, default=64)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=200)
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
        text_embedder=GloveTextEmbedding(device=device), batch_size=256),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_neighbors = [
    int(args.num_neighbors // 2**i) for i in range(args.num_layers)
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

model: Union[IDGNN, ContextGNN, ShallowRHSGNN]

model = ContextGNN(
    data=data, col_stats_dict=col_stats_dict,
    rhs_emb_mode=RHSEmbeddingMode.FUSION,
    dst_entity_table=task.dst_entity_table,
    num_nodes=num_dst_nodes_dict["train"], num_layers=args.num_layers,
    channels=args.channels, aggr="sum", norm="layer_norm", embedding_dim=64,
    torch_frame_model_kwargs={
        "channels": 128,
        "num_layers": 4,
    }, rhs_sample_size=1000).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0.0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)
    for batch in tqdm(loader_dict["train"], total=total_steps, desc="Train"):
        batch = batch.to(device)

        # Get ground-truth
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = sparse_tensor[input_id]

        # Optimization
        optimizer.zero_grad()

        logits, lhs_y_batch, rhs_y_index = model.forward_sample_softmax(
            batch, task.src_entity_table, task.dst_entity_table, src_batch,
            dst_index)
        edge_label_index = torch.stack([lhs_y_batch, rhs_y_index], dim=0)
        loss = sparse_cross_entropy(logits, edge_label_index)
        numel = len(batch[task.dst_entity_table].batch)
        loss.backward()

        optimizer.step()

        loss_accum += float(loss) * numel
        count_accum += numel

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    if count_accum == 0:
        warnings.warn(f"Did not sample a single '{task.dst_entity_table}' "
                      f"node in any mini-batch. Try to increase the number "
                      f"of layers/hops and re-try. If you run into memory "
                      f"issues with deeper nets, decrease the batch size.")

    return loss_accum / count_accum if count_accum > 0 else float("nan")


@torch.no_grad()
def test(loader: NeighborLoader, desc: str) -> np.ndarray:
    model.eval()

    pred_list: List[Tensor] = []
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)

        out = model(batch, task.src_entity_table,
                    task.dst_entity_table).detach()
        scores = torch.sigmoid(out)

        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


state_dict = None
best_val_metric = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(loader_dict["val"], desc="Val")
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
              f"Val metrics: {val_metrics}")

        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

assert state_dict is not None
model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"], desc="Best val")
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best val metrics: {val_metrics}")

test_pred = test(loader_dict["test"], desc="Test")
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
