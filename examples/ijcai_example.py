import argparse
import os.path as osp
import pickle
import warnings
from typing import Dict, List, Union

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
from contextgnn.nn.models import IDGNN, ContextGNN, ShallowRHSGNN
from contextgnn.utils import (
    RHSEmbeddingMode,
    calculate_hit_rate_ndcg,
    sparse_matrix_to_sparse_coo,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="contextgnn",
    choices=["contextgnn", "shallowrhsgnn"],
)
parser.add_argument("--lr", type=float, default=0.0068315852437584815)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=32)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--eval_k", type=int, default=10)
parser.add_argument("--gamma_rate", type=int, default=0.9564384518372194)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
seed_everything(args.seed)

src_entity_table = 'user'
dst_entity_table = 'item'
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
model: Union[IDGNN, ContextGNN, ShallowRHSGNN]

if args.model == "contextgnn":
    model = ContextGNN(
        data=data,
        col_stats_dict=col_stats_dict,
        rhs_emb_mode=RHSEmbeddingMode.FUSION,
        dst_entity_table=dst_entity_table,
        num_nodes=num_dst_nodes,
        num_layers=args.num_layers,
        channels=args.channels,
        aggr="sum",
        norm="batch_norm",
        embedding_dim=64,
        torch_frame_model_kwargs={
            "channels": 32,
            "num_layers": 4,
        },
    ).to(device)
elif args.model == 'shallowrhsgnn':
    model = ShallowRHSGNN(
        data=data,
        col_stats_dict=col_stats_dict,
        rhs_emb_mode=RHSEmbeddingMode.FUSION,
        dst_entity_table='item',
        num_nodes=num_dst_nodes,
        num_layers=args.num_layers,
        channels=args.channels,
        aggr="sum",
        norm="layer_norm",
        embedding_dim=64,
        torch_frame_model_kwargs={
            "channels": 128,
            "num_layers": 4,
        },
    ).to(device)
else:
    raise ValueError(f"Unsupported model type {args.model}.")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma_rate)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    sparse_tensor = SparseTensor(dst_nodes_dict["train"], device=device)
    for batch in tqdm(loader_dict["train"], total=total_steps, desc="Train"):
        batch = batch.to(device)

        # Get ground-truth
        input_id = batch['user'].input_id
        src_batch, dst_index = sparse_tensor[input_id]

        # Optimization
        optimizer.zero_grad()

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
        lr_scheduler.step()

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
def test(loader: NeighborLoader, desc: str, target=None) -> torch.Tensor:
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

                # We need to select 98 negative items.
                # It is too expensive to do isin on the entire set of items
                # Instead, we first quick sample 1000 items and find 98
                # negative items for this user. This works because the true
                # item set is very sparse.
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


state_dict = None
best_val_metric = 0
tune_metric = 'hr'

with open(osp.join(path, 'tst_int'), 'rb') as fs:
    target_list = pickle.load(fs)

for epoch in range(1, args.epochs + 1):
    train_loss = train()
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}")
    if epoch % 5 == 0:
        test_pred = test(loader_dict["test"], desc="Test", target=target_list)
        hr_10, ndcg_10 = calculate_hit_rate_ndcg(test_pred, target_list,
                                                 args.eval_k)
        print(f"Best test metrics on next best action: Hit Rate@{args.eval_k}:"
              f" {hr_10} NDCG@{args.eval_k}: {ndcg_10}")

test_pred = test(loader_dict["test"], desc="Test", target=target_list)
hr_1, _ = calculate_hit_rate_ndcg(test_pred, target_list, 1)
hr_5, ndcg_5 = calculate_hit_rate_ndcg(test_pred, target_list, 5)
hr_10, ndcg_10 = calculate_hit_rate_ndcg(test_pred, target_list, 10)
print(f"Best test metrics: HR@1 {hr_1} HR@5 {hr_5} HR@10 {hr_10} "
      f"ndcg_5 {ndcg_5} ndcg_10 {ndcg_10}")
