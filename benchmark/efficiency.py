"""Example script to run the models in this repository.

python relbench_example.py --dataset rel-trial --task site-sponsor-run
    --model contextgnn --epochs 10
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
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

warnings.filterwarnings("ignore", category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-trial")
parser.add_argument("--task", type=str, default="site-sponsor-run")
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
parser.add_argument("--cache_dir", type=str,
                    default=os.path.expanduser("~/.cache/relbench_examples"))
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

print("warming up...")
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

num_steps = 1_000
# num_uniq_batches = 20
# batches = []
# for i, batch in enumerate(loader_dict["train"]):
#     if i == num_uniq_batches:
#         break
#     batches.append(batch)


def create_model(model_type: str) -> Union[IDGNN, ContextGNN, ShallowRHSGNN]:
    if model_type == "idgnn":
        return IDGNN(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=args.num_layers,
            channels=args.channels,
            out_channels=1,
            aggr=args.aggr,
            norm="layer_norm",
            torch_frame_model_kwargs={
                "channels": 64,
                "num_layers": 4,
            },
        ).to(device)
    elif model_type == "contextgnn":
        return ContextGNN(
            data=data,
            col_stats_dict=col_stats_dict,
            rhs_emb_mode=RHSEmbeddingMode.FUSION,
            dst_entity_table=task.dst_entity_table,
            num_nodes=num_dst_nodes_dict["train"],
            num_layers=args.num_layers,
            channels=args.channels,
            aggr="sum",
            norm="layer_norm",
            embedding_dim=64,
            torch_frame_model_kwargs={
                "channels": 64,
                "num_layers": 4,
            },
        ).to(device)
    elif model_type == 'shallowrhsgnn':
        return ShallowRHSGNN(
            data=data,
            col_stats_dict=col_stats_dict,
            rhs_emb_mode=RHSEmbeddingMode.FUSION,
            dst_entity_table=task.dst_entity_table,
            num_nodes=num_dst_nodes_dict["train"],
            num_layers=args.num_layers,
            channels=args.channels,
            aggr="sum",
            norm="layer_norm",
            embedding_dim=64,
            torch_frame_model_kwargs={
                "channels": 64,
                "num_layers": 4,
            },
        ).to(device)
    raise ValueError(f"Unsupported model type {model_type}.")


sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)
for model_type in ["contextgnn", "idgnn", "shallowrhsgnn"]:
    model = create_model(model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train() -> float:
        model.train()

        print("warming up...")
        # for i, batch in batches[:10]:  # warmup
        for i, batch in enumerate(loader_dict["train"]):
            batch = batch.to(device)
            input_id = batch[task.src_entity_table].input_id
            src_batch, dst_index = sparse_tensor[input_id]
            optimizer.zero_grad()
            if model_type == 'idgnn':
                out = model(batch, task.src_entity_table,
                            task.dst_entity_table).flatten()
                batch_size = batch[task.src_entity_table].batch_size
                target = torch.isin(
                    batch[task.dst_entity_table].batch +
                    batch_size * batch[task.dst_entity_table].n_id,
                    src_batch + batch_size * dst_index,
                ).float()
                loss = F.binary_cross_entropy_with_logits(out, target)
            elif model_type in ['contextgnn', 'shallowrhsgnn']:
                logits = model(batch, task.src_entity_table, task.dst_entity_table)
                edge_label_index = torch.stack([src_batch, dst_index], dim=0)
                loss = sparse_cross_entropy(logits, edge_label_index)

            loss.backward()
            optimizer.step()

            if i == 9:
                break

        print("benchmarking...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # for i in range(num_steps):
        #     batch = batches[i % len(batches)]
        for i, batch in enumerate(loader_dict["train"]):
            batch = batch.to(device)
            input_id = batch[task.src_entity_table].input_id
            src_batch, dst_index = sparse_tensor[input_id]
            optimizer.zero_grad()
            if model_type == 'idgnn':
                out = model(batch, task.src_entity_table,
                            task.dst_entity_table).flatten()
                batch_size = batch[task.src_entity_table].batch_size
                target = torch.isin(
                    batch[task.dst_entity_table].batch +
                    batch_size * batch[task.dst_entity_table].n_id,
                    src_batch + batch_size * dst_index,
                ).float()
                loss = F.binary_cross_entropy_with_logits(out, target)
            elif model_type in ['contextgnn', 'shallowrhsgnn']:
                logits = model(batch, task.src_entity_table, task.dst_entity_table)
                edge_label_index = torch.stack([src_batch, dst_index], dim=0)
                loss = sparse_cross_entropy(logits, edge_label_index)

            loss.backward()
            optimizer.step()

            if i == num_steps - 1:
                print(f"done at {i}th step")
                break

        end.record()  # type: ignore
        torch.cuda.synchronize()
        gpu_time = start.elapsed_time(end)
        gpu_time_in_s = gpu_time / 1_000
        print(
            f"model: {model_type}, ",
            f"total: {gpu_time_in_s} s, "
            f"avg: {gpu_time_in_s / num_steps} s/iter, "
            f"avg: {num_steps / gpu_time_in_s} iter/s")

    train()


@torch.no_grad()
def test(loader: NeighborLoader, desc: str) -> np.ndarray:
    model.eval()

    pred_list: List[Tensor] = []
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        batch_size = batch[task.src_entity_table].batch_size

        if model_type == "idgnn":
            out = (model.forward(batch, task.src_entity_table,
                                 task.dst_entity_table).detach().flatten())
            scores = torch.zeros(batch_size, task.num_dst_nodes,
                                 device=out.device)
            scores[batch[task.dst_entity_table].batch,
                   batch[task.dst_entity_table].n_id] = torch.sigmoid(out)
        elif model_type in ['contextgnn', 'shallowrhsgnn']:
            out = model(batch, task.src_entity_table,
                        task.dst_entity_table).detach()
            scores = torch.sigmoid(out)
        else:
            raise ValueError(f"Unsupported model type: {model_type}.")

        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred
