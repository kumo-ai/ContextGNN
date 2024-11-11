"""Results running with the current setting on 48G memory machine:
Best test metrics: HR@1 0.4113 HR@5 0.6031 HR@10 0.6671
ndcg_5 0.5133189321177055 ndcg_10 0.5343443259121663.
"""
import argparse
import os
import os.path as osp
import pickle
import time
import warnings
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from relbench.modeling.loader import SparseTensor
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from torch_geometric.utils.map import map_index
from tqdm import tqdm

from contextgnn.data.ijcai_contest import IJCAI_Contest
from contextgnn.nn.models import ContextGNN, ShallowRHSGNN
from contextgnn.utils import (
    RHSEmbeddingMode,
    calculate_hit_rate_ndcg,
    sparse_matrix_to_sparse_coo,
)

TRAIN_CONFIG_KEYS = ["gamma_rate", "base_lr"]
TEST_LOSS_DELTA = 0.05

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="contextgnn",
    choices=["contextgnn", "shallowrhsgnn"],
)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--num_trials", type=int, default=10,
                    help="Number of Optuna-based hyper-parameter tuning.")
parser.add_argument(
    "--num_repeats", type=int, default=2,
    help="Number of repeated training and eval on the best config.")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--num_neighbors", type=int, default=64)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--eval_k", type=int, default=10)
parser.add_argument("--gamma_rate", type=int, default=0.95)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--result_path", default="results")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
seed_everything(args.seed)

# Constructing data
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '.data',
                'ijcai-contest')
behs = ['click', 'fav', 'cart', 'buy']
src_entity_table = 'user'
dst_entity_table = 'item'

with open(osp.join(path, 'tst_int'), 'rb') as fs:
    target_list = pickle.load(fs)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '.data',
                'ijcai-contest')
dataset = IJCAI_Contest(path)
data = dataset.datat
dst_nodes = dataset.dst_nodes
col_stats_dict = dataset.col_stats_dict
trnLabel = dataset.trnLabel

num_neighbors = [
    int(args.num_neighbors // 2**i) for i in range(args.num_layers)
]

model_cls: Type[Union[ContextGNN, ShallowRHSGNN]]

if args.model in ["contextgnn", "shallowrhsgnn"]:
    model_search_space = {
        "encoder_channels": [32, 64, 128],
        "encoder_layers": [2, 4],
        "channels": [32, 64, 128],
        "embedding_dim": [32, 64, 128],
        "norm": ["layer_norm", "batch_norm"],
        "rhs_emb_mode": [RHSEmbeddingMode.FUSION, RHSEmbeddingMode.LOOKUP]
    }
    train_search_space = {
        "base_lr": [0.001, 0.01],
        "gamma_rate": [0.8, 1.],
    }
    model_cls = (ContextGNN if args.model == "contextgnn" else ShallowRHSGNN)


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
          loader: NeighborLoader, train_sparse_tensor: SparseTensor) -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader), args.max_steps_per_epoch)
    for batch in tqdm(loader, total=total_steps, desc="Train"):
        batch = batch.to(device)

        # Get ground-truth
        input_id = batch['user'].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Optimization
        optimizer.zero_grad()

        # TODO: Fix idgnn part
        if args.model == 'idgnn':
            out = model(batch, src_entity_table, dst_entity_table).flatten()
            batch_size = batch[src_entity_table].batch_size

            # Get target label
            target = torch.isin(
                batch[dst_entity_table].batch +
                batch_size * batch[dst_entity_table].n_id,
                src_batch + batch_size * dst_index,
            ).float()

            loss = F.binary_cross_entropy_with_logits(out, target)
            numel = out.numel()
        elif args.model in ['contextgnn', 'shallowrhsgnn']:
            sampled_dst = torch.unique(dst_index)
            logits = model(batch, src_entity_table, dst_entity_table)
            logits = logits[:, sampled_dst]
            idx, _ = map_index(dst_index, sampled_dst, inclusive=True)
            edge_label_index = torch.stack([src_batch, idx], dim=0)
            loss = sparse_cross_entropy(logits, edge_label_index)
            numel = len(batch[dst_entity_table].batch)
        loss.backward()

        optimizer.step()

        loss_accum += float(loss) * numel
        count_accum += numel

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    if count_accum == 0:
        warnings.warn(f"Did not sample a single '{dst_entity_table}' "
                      f"node in any mini-batch. Try to increase the number "
                      f"of layers/hops and re-try. If you run into memory "
                      f"issues with deeper nets, decrease the batch size.")

    return loss_accum / count_accum if count_accum > 0 else float("nan")


trnLabel = sparse_matrix_to_sparse_coo(trnLabel).to(device)


@torch.no_grad()
def test(model: torch.nn.Module, loader: NeighborLoader, desc: str, stage: str,
         target=None) -> np.ndarray:
    model.eval()

    pred_list: List[Tensor] = []
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        batch_size = batch[src_entity_table].batch_size

        if args.model == "idgnn":
            out = (model.forward(batch, src_entity_table,
                                 dst_entity_table).detach().flatten())
            scores = torch.zeros(batch_size, num_dst_nodes, device=out.device)
            scores[batch[dst_entity_table].batch,
                   batch[dst_entity_table].n_id] = torch.sigmoid(out)
        elif args.model in ['contextgnn', 'shallowrhsgnn']:
            out = model(batch, src_entity_table, dst_entity_table).detach()
            scores = torch.sigmoid(out)
        else:
            raise ValueError(f"Unsupported model type: {args.model}.")

        # The neg set should be chosen from all sampled items.
        all_sampled_rhs = batch['item'].n_id.unique()
        num_sampled_rhs = len(all_sampled_rhs)

        if target is not None:
            # randomly select num_item indices
            batch_user = scores.shape[0]
            # full set is the sampled rhs set
            # you pick 99 items from the sampled rhs set.

            # you pick 99 items from the set where the user has never purchased

            all_sampled_rhs = batch['item'].n_id.unique()
            random_items = torch.randint(
                0, num_sampled_rhs, (batch_user, 100)).to(
                    scores.device)  # Shape: (batch_user, 100)
            for i in range(batch_size):
                user_idx = batch[src_entity_table].n_id[i]
                pos_item_per_user = trnLabel[user_idx].coalesce().indices(
                ).reshape(-1)

                indices = torch.randint(0, all_sampled_rhs.size(0), (1000, ))
                sampled_items = all_sampled_rhs[indices]

                random_items[i, :] = sampled_items[
                    ~torch.isin(sampled_items, pos_item_per_user)][:100]

                # include the target item if it is there
                target_item = target[user_idx]
                if target_item is not None:
                    random_items[i, -1] = target_item
            selected_scores = torch.gather(scores, 1, random_items)
            _, top_k_indices = torch.topk(
                selected_scores, args.eval_k, dim=1,
                sorted=True)  # Shape: (num_user, args.eval_k)
            pred_mini = random_items[
                torch.arange(random_items.size(0)).unsqueeze(1), top_k_indices]
        else:
            _, pred_mini = torch.topk(scores, k=args.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0)
    return pred


loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict = {}
split_date: Dict[str, int] = {}
split_date['train'] = 1112
split_date['test'] = 1112

num_src_nodes = data[src_entity_table].num_nodes
num_dst_nodes = data[dst_entity_table].num_nodes

for split in ["train", "test"]:
    dst_nodes_data = dst_nodes.data < split_date[split]
    dst_nodes_dict[split] = torch.sparse_coo_tensor(
        torch.stack([torch.tensor(dst_nodes.row),
                     torch.tensor(dst_nodes.col)]), dst_nodes_data,
        size=dst_nodes.shape).to_sparse_csr()
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=(src_entity_table,
                     torch.arange(num_src_nodes, dtype=torch.long)),
        input_time=torch.full((num_src_nodes, ), split_date[split],
                              dtype=torch.long),
        subgraph_type="bidirectional",
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def train_and_eval_with_cfg(
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    trial: Optional[optuna.trial.Trial] = None,
) -> float:
    if args.model in ["contextgnn", "shallowrhsgnn"]:
        model_cfg["num_nodes"] = num_dst_nodes
        model_cfg["dst_entity_table"] = dst_entity_table
    elif args.model == "idgnn":
        model_cfg["out_channels"] = 1
    encoder_model_kwargs = {
        "channels": model_cfg["encoder_channels"],
        "num_layers": model_cfg["encoder_layers"],
    }
    model_kwargs = {
        k: v
        for k, v in model_cfg.items()
        if k not in ["encoder_channels", "encoder_layers"]
    }
    # Use model_cfg to set up training procedure
    model = model_cls(
        **model_kwargs,
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=args.num_layers,
        torch_frame_model_kwargs=encoder_model_kwargs,
    ).to(device)
    model.reset_parameters()
    # Use train_cfg to set up training procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["base_lr"])
    lr_scheduler = ExponentialLR(optimizer, gamma=train_cfg["gamma_rate"])

    for epoch in range(1, args.epochs + 1):
        train_sparse_tensor = SparseTensor(dst_nodes_dict["train"],
                                           device=device)
        try:
            train_loss = train(model, optimizer, loader_dict["train"],
                               train_sparse_tensor)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("CUDA out of memory error. Clearing cache...")
                torch.cuda.empty_cache()  # Clear the cache to free up memory
                return 0.0
            else:
                raise e
        optimizer.zero_grad()
        lr_scheduler.step()
        print(f"Train Loss: {train_loss:.4f}")
    test_pred = test(model, loader_dict["test"], "test", "test",
                     target=target_list)
    hr_1, _ = calculate_hit_rate_ndcg(test_pred, target_list, 1)
    hr_5, ndcg_5 = calculate_hit_rate_ndcg(test_pred, target_list, 5)
    hr_10, ndcg_10 = calculate_hit_rate_ndcg(test_pred, target_list, 10)
    print(f"Best test metrics: HR@1 {hr_1} HR@5 {hr_5} HR@10 {hr_10} "
          f"ndcg_5 {ndcg_5} ndcg_10 {ndcg_10}")
    return ndcg_10


def objective(trial: optuna.trial.Trial) -> float:
    model_cfg: Dict[str, Any] = {}
    for name, search_list in model_search_space.items():
        assert isinstance(search_list, list)
        model_cfg[name] = trial.suggest_categorical(name, search_list)
    train_cfg: Dict[str, Any] = {}
    for name, search_list in train_search_space.items():
        assert isinstance(search_list, list)
        if name == "batch_size":
            train_cfg[name] = trial.suggest_categorical(name, search_list)
        else:
            train_cfg[name] = trial.suggest_loguniform(name, search_list[0],
                                                       search_list[1])

    best_test_metric = train_and_eval_with_cfg(model_cfg=model_cfg,
                                               train_cfg=train_cfg,
                                               trial=trial)
    return best_test_metric


def main_gnn() -> None:
    # Hyper-parameter optimization with Optuna
    print("Hyper-parameter search via Optuna")
    start_time = time.time()
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize",
    )
    study.optimize(objective, n_trials=args.num_trials)
    end_time = time.time()
    search_time = end_time - start_time
    print("Hyper-parameter search done. Found the best config.")
    params = study.best_params
    best_train_cfg = {}
    for train_cfg_key in TRAIN_CONFIG_KEYS:
        best_train_cfg[train_cfg_key] = params.pop(train_cfg_key)
    best_model_cfg = params

    print(f"Repeat experiments {args.num_repeats} times with the best train "
          f"config {best_train_cfg} and model config {best_model_cfg}.")
    start_time = time.time()
    best_test_metrics = []
    for _ in range(args.num_repeats):
        best_test_metric = train_and_eval_with_cfg(best_model_cfg,
                                                   best_train_cfg)
        best_test_metrics.append(best_test_metric)
    end_time = time.time()
    final_model_time = (end_time - start_time) / args.num_repeats
    best_test_metrics_array = np.array(best_test_metrics)

    result_dict = {
        "args": args.__dict__,
        "best_test_metrics": best_test_metrics_array,
        "best_test_metric": best_test_metrics_array.mean(),
        "best_train_cfg": best_train_cfg,
        "best_model_cfg": best_model_cfg,
        "search_time": search_time,
        "final_model_time": final_model_time,
        "total_time": search_time + final_model_time,
    }
    print(result_dict)
    # Save results
    if args.result_path != "":
        os.makedirs(args.result_path, exist_ok=True)
        torch.save(result_dict,
                   osp.join(args.result_path, f"ijcai_{args.model}"))


if __name__ == "__main__":
    print(args)
    main_gnn()
