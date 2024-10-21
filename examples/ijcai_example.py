import argparse
import os.path as osp
import pickle
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from relbench.modeling.loader import SparseTensor
from torch import Tensor
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from tqdm import tqdm

from contextgnn.nn.models import IDGNN, ContextGNN, ShallowRHSGNN
from contextgnn.utils import RHSEmbeddingMode

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="contextgnn",
    choices=["contextgnn", "idgnn", "shallowrhsgnn"],
)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--max_steps_per_epoch", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--eval_k", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
seed_everything(args.seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'ijcai-contest')
behs = ['click', 'fav', 'cart', 'buy']
src_entity_table = 'user'
dst_entity_table = 'item'

data = HeteroData()


def calculate_hit_rate(pred: torch.Tensor, target: List[Optional[int]]):
    r"""Calculates hit rate when pred is a tensor and target is a list
    Args:
        pred (torch.Tensor): Prediction tensor of size (num_entity,
            num_target_predicitons_per_entity).
        target (List[int]): A list of shape num_entity, where the
            value is None if user doesn't have a next best action.
            The value is the dst node id if there is a next best
            action.
    """
    hits = 0
    total = 0
    for i in range(len(target)):
        if target[i] is not None:
            total += 1
            if target[i] in pred[i]:
                hits += 1

    return hits / total


def calculate_hit_rate_on_sparse_target(pred: torch.Tensor,
                                        target: torch.sparse.Tensor):
    r"""Calculates hit rate when pred is a tensor and target is a sparse
    tensor
    Args:
        pred (torch.Tensor): Prediction tensor of size (num_entity,
            num_target_predicitons_per_entity).
        target (torch.sparse.Tensor): Target sparse tensor.
    """
    crow_indices = target.crow_indices()
    col_indices = target.col_indices()
    values = target.values()
    # Iterate through each row and check if predictions match ground truth
    hits = 0
    num_rows = val_pred.shape[0]

    for i in range(num_rows):
        # Get the ground truth indices for this row
        row_start = crow_indices[i].item()
        row_end = crow_indices[i + 1].item()
        dst_indices = col_indices[row_start:row_end]
        bool_indices = values[row_start:row_end]
        true_indices = dst_indices[bool_indices].tolist()

        # Check if any of the predicted values match the true indices
        pred_indices = pred[i]
        if torch.isin(pred_indices, true_indices).any():
            hits += 1

    # Callculate hit rate
    hit_rate = hits / num_rows
    return hit_rate


def create_edge(data, behavior, beh_idx, pkey_name, pkey_idx):
    # fkey -> pkey edges
    edge_index = torch.stack([beh_idx, pkey_idx], dim=0)
    edge_type = (behavior, f"f2p_{behavior}", pkey_name)
    data[edge_type].edge_index = sort_edge_index(edge_index)

    # pkey -> fkey edges.
    # "rev_" is added so that PyG loader recognizes the reverse edges
    edge_index = torch.stack([pkey_idx, beh_idx], dim=0)
    edge_type = (pkey_name, f"rev_f2p_{behavior}", behavior)
    data[edge_type].edge_index = sort_edge_index(edge_index)


col_stats_dict = {}
dst_nodes = None
for i in range(len(behs)):
    behavior = behs[i]
    with open(osp.join(path, 'trn_' + behavior), 'rb') as fs:
        mat = pickle.load(fs)
    if i == 0:
        dataset = Dataset(pd.DataFrame({"__const__": np.ones(mat.shape[0])}),
                          col_to_stype={"__const__": stype.numerical})
        dataset.materialize()
        data['user'].tf = dataset.tensor_frame
        col_stats_dict['user'] = dataset.col_stats
        dataset = Dataset(pd.DataFrame({"__const__": np.ones(mat.shape[1])}),
                          col_to_stype={"__const__": stype.numerical})
        dataset.materialize()
        data['item'].tf = dataset.tensor_frame
        col_stats_dict['item'] = dataset.col_stats
        data['user'].n_id = torch.arange(mat.shape[0], dtype=torch.long)
        data['item'].n_id = torch.arange(mat.shape[1], dtype=torch.long)
    col_to_stype = {"time": stype.timestamp}
    dataset = Dataset(pd.DataFrame({'time': mat.data}),
                      col_to_stype=col_to_stype)
    dataset.materialize()
    col_stats_dict[behavior] = dataset.col_stats
    data[behavior].tf = dataset.tensor_frame
    data[behavior].time = torch.tensor(mat.data, dtype=torch.long)
    data[behavior].x = torch.tensor(np.ones(len(mat.data))).view(-1, 1)
    data[behavior].n_id = torch.arange(len(mat.data), dtype=torch.long)
    coo_mat = sp.coo_matrix(mat)
    if behavior == 'buy':
        dst_nodes = coo_mat
    beh_idx = torch.arange(len(coo_mat.data), dtype=torch.long)
    create_edge(data, behavior, beh_idx, src_entity_table,
                torch.tensor(coo_mat.row, dtype=torch.long))
    create_edge(data, behavior, beh_idx, dst_entity_table,
                torch.tensor(coo_mat.col, dtype=torch.long))

num_neighbors = [
    int(args.num_neighbors // 2**i) for i in range(args.num_layers)
]

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict = {}
split_date: Dict[str, int] = {}
split_date['train'] = 1103
split_date['val'] = 1110
split_date['test'] = 1111

num_src_nodes = data[src_entity_table].num_nodes
num_dst_nodes = data[dst_entity_table].num_nodes

for split in ["train", "val", "test"]:
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

if args.model == "idgnn":
    model = IDGNN(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=args.num_layers,
        channels=args.channels,
        out_channels=1,
        aggr=args.aggr,
        norm="layer_norm",
        torch_frame_model_kwargs={
            "channels": 128,
            "num_layers": 4,
        },
    ).to(device)
elif args.model == "contextgnn":
    model = ContextGNN(
        data=data,
        col_stats_dict=col_stats_dict,
        rhs_emb_mode=RHSEmbeddingMode.FUSION,
        dst_entity_table=dst_entity_table,
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
            logits = model(batch, src_entity_table, dst_entity_table)
            edge_label_index = torch.stack([src_batch, dst_index], dim=0)
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


@torch.no_grad()
def test(loader: NeighborLoader, desc: str) -> np.ndarray:
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

        _, pred_mini = torch.topk(scores, k=args.eval_k, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


state_dict = None
best_val_metric = 0
tune_metric = 'hr'
val_metrics = dict()
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(loader_dict["val"], desc="Val")
        val_metrics[tune_metric] = calculate_hit_rate_on_sparse_target(
            val_pred, dst_nodes_dict['val'])
        print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
              f"Val metrics: {val_metrics}")

        if val_metrics[tune_metric] > best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

assert state_dict is not None
model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"], desc="Best val")
val_metrics = calculate_hit_rate_on_sparse_target(val_pred,
                                                  dst_nodes_dict['val'])
print(f"Best val metrics: {val_metrics}")

with open(osp.join(path, 'tst_int'), 'rb') as fs:
    mat = pickle.load(fs)

test_pred = test(loader_dict["test"], desc="Test")
test_metrics = calculate_hit_rate(test_pred, mat)
print(f"Best test metrics: {test_metrics}")
