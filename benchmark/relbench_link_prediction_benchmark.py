import argparse
import json
import os
import os.path as osp
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import optuna
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
from torch.optim.lr_scheduler import ExponentialLR
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from tqdm import tqdm

from hybridgnn.nn.models import IDGNN, HybridGNN
from hybridgnn.utils import GloveTextEmbedding
from torch.utils.tensorboard import SummaryWriter


LINK_PREDICTION_METRIC = "link_prediction_map"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-trial")
parser.add_argument("--task", type=str, default="condition-sponsor-run")
parser.add_argument(
    "--model",
    type=str,
    default="hybridgnn",
    choices=["hybridgnn", "idgnn"],
)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--num_trials", type=int, default=10,
                    help="Number of Optuna-based hyper-parameter tuning.")
parser.add_argument(
    "--num_repeats", type=int, default=1,
    help="Number of repeated training and eval on the best config.")
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last",
                    choices=["last", "uniform"])
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache_dir", type=str,
                    default=os.path.expanduser("~/.cache/relbench_examples"))
parser.add_argument("--result_path", type=str, default="result.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)

seed_everything(args.seed)

if args.dataset == "rel-trial":
    args.num_layers = 4

dataset: Dataset = get_dataset(args.dataset, download=True)
task: RecommendationTask = get_task(args.dataset, args.task, download=True)
tune_metric = LINK_PREDICTION_METRIC
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
        text_embedder=GloveTextEmbedding(device=device), batch_size=512),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_neighbors = [
    int(args.num_neighbors // 2**i) for i in range(args.num_layers)
]

model_cls: Type[Union[IDGNN, HybridGNN]]

if args.model == "idgnn":
    search_space = {
        "channels": [64, 128, 256],
        "norm": ["layer_norm", "batch_norm"],
        "batch_size": [256, 512, 1024],
        "base_lr": [0.0001, 0.01],
        "gamma_rate": [0.9, 0.95, 1.],
    }
    model_cls = IDGNN
elif args.model == "hybridgnn":
    search_space = {
        # "channels": [64, 128, 256],
        "channels": [32, 64, 128],  # 128 in kumo hm.py
        # "embedding_dim": [64, 128, 256],
        "embedding_dim": [32],  # 32 in kumo hm.py
        "norm": ["layer_norm", "batch_norm"],
        # "batch_size": [256, 512, 1024],
        "batch_size": [64, 128],  # 256],
        "base_lr": [0.001, 0.01],
        "gamma_rate": [0.9, 0.95, 1.],
    }
    model_cls = HybridGNN


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: NeighborLoader,
    train_sparse_tensor: SparseTensor,
) -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader), args.max_steps_per_epoch)
    for batch in tqdm(loader, total=total_steps, desc="Train"):
        batch = batch.to(device)

        # Get ground-truth
        input_id = batch[task.src_entity_table].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Optimization
        optimizer.zero_grad()

        if args.model == "idgnn":
            out = model(batch, task.src_entity_table,
                        task.dst_entity_table).flatten()
            batch_size = batch[task.src_entity_table].batch_size

            # Get target label
            target = torch.isin(
                batch[task.dst_entity_table].batch +
                batch_size * batch[task.dst_entity_table].n_id,
                src_batch + batch_size * dst_index,
            ).float()

            loss = F.binary_cross_entropy_with_logits(out, target)
            numel = out.numel()
        else:
            logits = model(batch, task.src_entity_table, task.dst_entity_table)
            edge_label_index = torch.stack([src_batch, dst_index], dim=0)
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
def test(model: torch.nn.Module, loader: NeighborLoader, stage: str) -> float:
    model.eval()

    pred_list: List[Tensor] = []
    for batch in tqdm(loader, desc=stage):
        batch = batch.to(device)
        batch_size = batch[task.src_entity_table].batch_size

        if args.model == "idgnn":
            out = (model.forward(batch, task.src_entity_table,
                                 task.dst_entity_table).detach().flatten())
            scores = torch.zeros(batch_size, task.num_dst_nodes,
                                 device=out.device)
            scores[batch[task.dst_entity_table].batch,
                   batch[task.dst_entity_table].n_id] = torch.sigmoid(out)
        elif args.model == "hybridgnn":
            # Get ground-truth
            out = model(batch, task.src_entity_table,
                        task.dst_entity_table).detach()
            scores = torch.sigmoid(out)
        else:
            raise ValueError(f"Unsupported model type: {args.model}.")

        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)

    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    res = task.evaluate(pred, task.get_table(stage))
    return res[LINK_PREDICTION_METRIC]


def train_and_eval_with_cfg(
    cfg: Dict[str, Any],
    trial: Optional[optuna.trial.Trial] = None,
) -> Tuple[float, float]:
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
            batch_size=cfg["batch_size"],
            temporal_strategy=args.temporal_strategy,
            shuffle=split == "train",
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

    if args.model == "hybridgnn":
        model = model_cls(
            channels=cfg["channels"],
            norm=cfg["norm"],
            num_nodes=num_dst_nodes_dict["train"],
            embedding_dim=cfg["embedding_dim"],
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=args.num_layers,
        ).to(device)
    elif args.model == "idgnn":
        model = model_cls(
            channels=cfg["channels"],
            norm=cfg["norm"],
            out_channels=1,
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=args.num_layers,
        ).to(device)

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["base_lr"])
    lr_scheduler = ExponentialLR(optimizer, gamma=cfg["gamma_rate"])

    best_val_metric: float = 0.0
    best_test_metric: float = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model,
            optimizer,
            loader_dict["train"],
            SparseTensor(dst_nodes_dict["train"][1], device=device),
        )
        optimizer.zero_grad()
        val_metric = test(model, loader_dict["val"], "val")

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_test_metric = test(model, loader_dict["test"], "test")

        lr_scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}")

        if trial is not None:
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    print(
        f"Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}")
    return best_val_metric, best_test_metric


def objective(trial: optuna.trial.Trial) -> float:
    cfg = {key: trial.suggest_categorical(key, values) for key, values in search_space.items()}
    print(cfg)
    run_name = f"exp_{'_'.join([str(k) + '_' + str(v) for k, v in cfg.items()])}"
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_hparams(cfg, {})

    best_val_metric, _ = train_and_eval_with_cfg(cfg, trial=trial)

    writer.add_scalar('val_metric', best_val_metric)
    writer.add_hparams(
        cfg,
        {'val_metric': best_val_metric}
    )
    writer.close()
    return best_val_metric


def main_gnn() -> None:
    # Hyper-parameter optimization with Optuna
    print("Hyper-parameter search via Optuna")
    # start_time = time.time()
    # study = optuna.create_study(
    #     pruner=optuna.pruners.MedianPruner(),
    #     sampler=optuna.samplers.GridSampler(search_space=search_space),
    #     direction="maximize",
    # )
    # study.optimize(objective, n_trials=args.num_trials)
    # end_time = time.time()
    # best_cfg = study.best_params
    best_cfg =  {'channels': 128, 'embedding_dim': 32, 'norm': 'layer_norm', 'batch_size': 64, 'base_lr': 0.01, 'gamma_rate': 0.95}
    search_time = end_time - start_time
    print("Hyper-parameter search done. Found the best config.")

    print(f"Repeat experiments {args.num_repeats} times with the best config "
          f"config {best_cfg}.")
    start_time = time.time()
    best_val_metrics = []
    best_test_metrics = []
    for _ in range(args.num_repeats):
        best_val_metric, best_test_metric = train_and_eval_with_cfg(best_cfg)
        best_val_metrics.append(best_val_metric)
        best_test_metrics.append(best_test_metric)
    end_time = time.time()
    final_model_time = (end_time - start_time) / args.num_repeats
    best_val_metrics_array = np.array(best_val_metrics)
    best_test_metrics_array = np.array(best_test_metrics)

    result_dict = {
        "args": args.__dict__,
        "best_val_metrics": best_val_metrics_array,
        "best_test_metrics": best_test_metrics_array,
        "best_val_metric": best_val_metrics_array.mean(),
        "best_test_metric": best_test_metrics_array.mean(),
        "best_cfg": best_cfg,
        "search_time": search_time,
        "final_model_time": final_model_time,
    }
    print(result_dict)
    if args.result_path != "":
        os.makedirs(args.result_path, exist_ok=True)
        torch.save(
            result_dict,
            osp.join(args.result_path,
                     f"{args.dataset}_{args.task}_{args.model}"))


if __name__ == "__main__":
    print(args)
    main_gnn()

# === hybridgnn ===
# {'channels': 64, 'embedding_dim': 32, 'norm': 'batch_norm', 'batch_size': 128, 'base_lr': 0.001, 'gamma_rate': 0.9}
# 13.337 GiB during testing

# {'channels': 64, 'embedding_dim': 32, 'norm': 'batch_norm', 'batch_size': 128, 'base_lr': 0.001, 'gamma_rate': 1.0}
# OOM

# {'channels': 64, 'embedding_dim': 32, 'norm': 'batch_norm', 'batch_size': 64, 'base_lr': 0.01, 'gamma_rate': 0.95}
# 14.548 GiB during testing

# {'channels': 64, 'embedding_dim': 32, 'norm': 'batch_norm', 'batch_size': 128, 'base_lr': 0.01, 'gamma_rate': 1.0}
# 14.552 GiB during testing

# {'channels': 64, 'embedding_dim': 32, 'norm': 'batch_norm', 'batch_size': 128, 'base_lr': 0.001, 'gamma_rate': 1.0}
# 14.552 GiB during testing

# Repeat experiments 1 times with the best config config {'channels': 128, 'embedding_dim': 32, 'norm': 'layer_norm', 'batch_size': 64, 'base_lr': 0.01, 'gamma_rate': 0.95}.
# [1]    2077555 killed     python benchmark/relbench_link_prediction_benchmark.py --dataset rel-avito
