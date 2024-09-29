"""Example script to run the models in this repository.

python static_example.py
"""

import argparse
import os
import os.path as osp
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from relbench.metrics import (
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
)
from relbench.modeling.graph import LinkTrainTableInput
from relbench.modeling.loader import SparseTensor
from relbench.modeling.utils import to_unix_time
from torch import Tensor
from torch_frame import stype
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_frame.utils import infer_df_stype
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from tqdm import tqdm

from hybridgnn.nn.models import IDGNN, HybridGNN, ShallowRHSGNN
from hybridgnn.utils import RHSEmbeddingMode

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="yelp2018",
                    choices=["yelp2018", "amazon-book", "gowalla"])
parser.add_argument(
    "--model",
    type=str,
    default="hybridgnn",
    choices=["hybridgnn", "idgnn", "shallowrhsgnn"],
)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--supervision_ratio", type=float, default=0.5)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--num_neighbors", type=int, default=32)
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

PSEUDO_TIME = "pseudo_time"
TRAIN_SET_TIMESTAMP = pd.Timestamp("1970-01-01")
SRC_ENTITY_TABLE = "user_table"
DST_ENTITY_TABLE = "item_table"
TRANSACTION_TABLE = "transaction_table"
SRC_ENTITY_COL = "user_id"
DST_ENTITY_COL = "item_id"
EVAL_K = 20
METRICS = [
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
]

dataset = args.dataset
input_data_dir = osp.join("../", "static_data", dataset)

# Load user data
user_path = osp.join(input_data_dir, "user_list.txt")
src_df = pd.read_csv(user_path, delim_whitespace=True)
# Drop `org_id` and rename `remap_id` to `user_id`
src_df = src_df.drop(columns=['org_id']).rename(
    columns={'remap_id': SRC_ENTITY_COL})
src_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP

# Load item data
item_path = osp.join(input_data_dir, "item_list.txt")
dst_df = pd.read_csv(item_path, delim_whitespace=True)
# Drop `org_id` and rename `remap_id` to `item_id`
dst_df = dst_df.drop(columns=['org_id']).rename(
    columns={'remap_id': DST_ENTITY_COL})
dst_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP
NUM_DST_NODES = len(dst_df)

# Load user item link for train data
train_path = osp.join(input_data_dir, "train.txt")
user_ids = []
item_ids = []
with open(train_path, 'r') as file:
    for line in file:
        values = list(map(int, line.split()))
        user_id = values[0]
        item_ids_for_user = values[1:]
        user_ids.append(user_id)
        item_ids.append(item_ids_for_user)
train_df = pd.DataFrame({SRC_ENTITY_COL: user_ids, DST_ENTITY_COL: item_ids})
# Shuffle train data
train_df = train_df.sample(frac=1,
                           random_state=args.seed).reset_index(drop=True)
# Add pseudo time column
train_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP

# load user item link for test data
test_path = osp.join(input_data_dir, "test.txt")
user_ids = []
item_ids = []
with open(test_path, 'r') as file:
    for line in file:
        values = list(map(int, line.split()))
        user_id = values[0]
        item_ids_for_user = values[1:]
        user_ids.append(user_id)
        item_ids.append(item_ids_for_user)
test_df = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids})
# Shuffle train data
test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
# Add pseudo time column
test_df[PSEUDO_TIME] = TRAIN_SET_TIMESTAMP + pd.Timedelta(days=1)

train_df_explode = train_df.explode(DST_ENTITY_COL).reset_index(drop=True)
target_df = train_df_explode

table_dict = {
    SRC_ENTITY_TABLE: src_df,
    DST_ENTITY_TABLE: dst_df,
    TRANSACTION_TABLE: target_df,
}


def get_static_stype_proposal(
        table_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, stype]]:
    r"""Infer style for table columns."""
    inferred_col_to_stype_dict = {}
    for table_name, df in table_dict.items():
        df = df.sample(min(1_000, len(df)))
        inferred_col_to_stype = infer_df_stype(df)
        inferred_col_to_stype_dict[table_name] = inferred_col_to_stype
    return inferred_col_to_stype_dict


col_to_stype_dict = get_static_stype_proposal(table_dict)


def static_data_make_pkey_fkey_graph(
    table_dict: Dict[str, pd.DataFrame],
    col_to_stype_dict: Dict[str, Dict[str, stype]],
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    data = HeteroData()
    col_stats_dict = dict()

    # Update src nodes information in HeteroData and col_stats_dict
    src_col_to_stype = col_to_stype_dict[SRC_ENTITY_TABLE]
    src_dataset = Dataset(
        df=table_dict[SRC_ENTITY_TABLE],
        col_to_stype=src_col_to_stype,
    ).materialize()
    data[SRC_ENTITY_TABLE].tf = src_dataset.tensor_frame
    data[SRC_ENTITY_TABLE].time = torch.from_numpy(
        to_unix_time(table_dict[SRC_ENTITY_TABLE][PSEUDO_TIME]))
    col_stats_dict[SRC_ENTITY_TABLE] = src_dataset.col_stats

    # Update dst nodes information in HeteroData and col_stats_dict
    dst_col_to_stype = col_to_stype_dict[DST_ENTITY_TABLE]
    dst_dataset = Dataset(
        df=table_dict[DST_ENTITY_TABLE],
        col_to_stype=dst_col_to_stype,
    ).materialize()
    data[DST_ENTITY_TABLE].tf = dst_dataset.tensor_frame
    data[DST_ENTITY_TABLE].time = torch.from_numpy(
        to_unix_time(table_dict[DST_ENTITY_TABLE][PSEUDO_TIME]))
    col_stats_dict[DST_ENTITY_TABLE] = dst_dataset.col_stats

    fkey_index = torch.from_numpy(
        table_dict[TRANSACTION_TABLE][SRC_ENTITY_COL].astype(int).values)
    pkey_index = torch.from_numpy(
        table_dict[TRANSACTION_TABLE][DST_ENTITY_COL].astype(int).values)
    edge_index = torch.stack([fkey_index, pkey_index], dim=0)
    edge_type = (SRC_ENTITY_TABLE, SRC_ENTITY_COL, DST_ENTITY_TABLE)
    data[edge_type].edge_index = sort_edge_index(edge_index)

    reverse_edge_index = torch.stack([pkey_index, fkey_index], dim=0)
    reverse_edge_type = (DST_ENTITY_TABLE, DST_ENTITY_COL, SRC_ENTITY_TABLE)
    data[reverse_edge_type].edge_index = sort_edge_index(reverse_edge_index)
    data.validate()
    return data, col_stats_dict


data, col_stats_dict = static_data_make_pkey_fkey_graph(
    table_dict=table_dict,
    col_to_stype_dict=col_to_stype_dict,
)

# This setup is giving good test performance with args.epochs == 1
num_neighbors = [
    8,
    8,
    8,
]
# num_neighbors = [
#     int(args.num_neighbors // 2**i) for i in range(args.num_layers)
# ]


def static_get_link_train_table_input(
    transaction_df: pd.DataFrame,
    num_dst_nodes: int,
) -> LinkTrainTableInput:
    df = transaction_df
    src_node_idx: Tensor = torch.from_numpy(
        df[SRC_ENTITY_COL].astype(int).values)
    exploded = df[DST_ENTITY_COL].explode().dropna()

    coo_indices = torch.from_numpy(
        np.stack([exploded.index.values,
                  exploded.values.astype(int)]))
    sparse_coo = torch.sparse_coo_tensor(
        coo_indices,
        torch.ones(coo_indices.size(1), dtype=bool),  # type: ignore
        (len(src_node_idx), num_dst_nodes),
    )
    dst_node_indices = sparse_coo.to_sparse_csr()
    time = torch.from_numpy(to_unix_time(df[PSEUDO_TIME]))
    return LinkTrainTableInput(
        src_nodes=(SRC_ENTITY_TABLE, src_node_idx),
        dst_nodes=(DST_ENTITY_TABLE, dst_node_indices),
        num_dst_nodes=num_dst_nodes,
        src_time=time,
    )


loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
num_dst_nodes_dict: Dict[str, int] = {}
for split in ["train", "test"]:
    if split == "train":
        table = train_df
    elif split == "test":
        table = test_df
    table_input = static_get_link_train_table_input(
        table, num_dst_nodes=NUM_DST_NODES)
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
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        disjoint=True,
    )

model: Union[IDGNN, HybridGNN, ShallowRHSGNN]
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
            "num_layers": args.num_layers,
        },
        is_static=True,
    ).to(device)
elif args.model == "hybridgnn":
    model = HybridGNN(
        data=data,
        col_stats_dict=col_stats_dict,
        rhs_emb_mode=RHSEmbeddingMode.FUSION,
        dst_entity_table=DST_ENTITY_TABLE,
        num_nodes=num_dst_nodes_dict["train"],
        num_layers=args.num_layers,
        channels=args.channels,
        aggr="sum",
        norm="layer_norm",
        embedding_dim=64,
        torch_frame_model_kwargs={
            "channels": 128,
            "num_layers": args.num_layers,
        },
        is_static=True,
    ).to(device)
elif args.model == 'shallowrhsgnn':
    model = ShallowRHSGNN(
        data=data,
        col_stats_dict=col_stats_dict,
        rhs_emb_mode=RHSEmbeddingMode.FUSION,
        dst_entity_table=DST_ENTITY_TABLE,
        num_nodes=num_dst_nodes_dict["train"],
        num_layers=args.num_layers,
        channels=args.channels,
        aggr="sum",
        norm="layer_norm",
        embedding_dim=64,
        torch_frame_model_kwargs={
            "channels": 128,
            "num_layers": args.num_layers,
        },
        is_static=True,
    ).to(device)
else:
    raise ValueError(f"Unsupported model type {args.model}.")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)

    sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)
    for batch in tqdm(loader_dict["train"], total=total_steps, desc="Train"):
        batch = batch.to(device)

        # Get ground truth edges
        input_id = batch[SRC_ENTITY_TABLE].input_id
        src_batch, dst_index = sparse_tensor[input_id]
        edge_label_index = torch.stack([src_batch, dst_index], dim=0)

        train_seed_nodes = loader_dict["train"].input_nodes[1].to(
            src_batch.device)
        global_src_index = train_seed_nodes[
            batch[SRC_ENTITY_TABLE].input_id[src_batch]]
        global_edge_label_index = torch.stack([global_src_index, dst_index],
                                              dim=0)

        # batch.edge_type=[
        # ('user_table', 'user_id', 'item_table'),
        # ('item_table', 'item_id', 'user_table'),
        # ]
        edge_type = (SRC_ENTITY_TABLE, SRC_ENTITY_COL, DST_ENTITY_TABLE)
        edge_index = batch[edge_type].edge_index

        # NOTE: Assume that dst node indices are consecutive
        # starting from 0 and monotonically increasing, which is
        # true for all 3 static datasets: amazon-book, gowalla and
        # yelp2018 that we have.
        global_src_index = batch[SRC_ENTITY_TABLE].n_id[edge_index[0]]
        global_dst_index = batch[DST_ENTITY_TABLE].n_id[edge_index[1]]
        global_edge_index = torch.stack([global_src_index, global_dst_index])

        global_src_batch = batch[SRC_ENTITY_TABLE].batch[edge_index[0]]
        global_dst_batch = batch[DST_ENTITY_TABLE].batch[edge_index[1]]
        assert all(global_src_batch == global_dst_batch) is True

        edge_index_message_passing_list = []
        edge_label_index_list = []

        # With a batch for each disjoint subgraph, we first extract all the
        # ground truth labels for input seed nodes. Then we sample the ground
        # truth labels according to args.supervision_ratio as supervision
        # edges for this particular disjoint subgraph. Next we will remove
        # the selected supervision edges from edge_index to get the message
        # passing edges to avoid data leakage.
        for subgraph_index in range(args.batch_size):
            subgraph_mask = (global_src_batch == subgraph_index)

            # Extract ground truth links for the src nodes in this subgraph.
            subgraph_edge_index = edge_index[:, subgraph_mask]
            subgraph_global_edge_index = global_edge_index[:, subgraph_mask]
            subgraph_src_indices = subgraph_global_edge_index[0]
            subgraph_src_indices_unique = torch.unique(subgraph_src_indices)

            ground_truth_mask = torch.isin(global_edge_label_index[0],
                                           subgraph_src_indices_unique)
            subgraph_global_edge_label_index = (
                global_edge_label_index[:, ground_truth_mask])
            subgraph_edge_label_index = edge_label_index[:, ground_truth_mask]

            # Create supervision edges for this subgraph.
            supervision_edges_sample_size = (int(
                subgraph_global_edge_label_index.shape[1] *
                args.supervision_ratio))
            subgraph_sample_indices = torch.randperm(
                subgraph_global_edge_label_index.shape[1],
                device=subgraph_global_edge_label_index.device,
            )[:supervision_edges_sample_size]

            # Remove supervision edges from message passing ones.
            subgraph_global_edge_label_index_sample = (
                subgraph_global_edge_label_index[:, subgraph_sample_indices])
            subgraph_edge_label_index_sample = (
                subgraph_edge_label_index[:, subgraph_sample_indices])
            subgraph_global_edge_label_index_hash = (
                subgraph_global_edge_label_index_sample[0, :] * NUM_DST_NODES +
                subgraph_global_edge_label_index_sample[1, :])
            subgraph_global_edge_index_hash = subgraph_global_edge_index[
                0, :] * NUM_DST_NODES + subgraph_global_edge_index[1, :]
            subgraph_supervison_edge_mask = ~torch.isin(
                subgraph_global_edge_index_hash,
                subgraph_global_edge_label_index_hash,
            )
            subgraph_edge_index_message_passing = (
                subgraph_edge_index[:, subgraph_supervison_edge_mask])
            edge_index_message_passing_list.append(
                subgraph_edge_index_message_passing)
            edge_label_index_list.append(subgraph_edge_label_index_sample)

        # Reconstruct message passing and supervision edges for a batch.
        edge_index_message_passing_concat = torch.cat(
            [x for x in edge_index_message_passing_list],
            dim=1,
        )
        edge_label_index_concat = torch.cat(
            [x for x in edge_label_index_list],
            dim=1,
        )
        edge_label_index_concat_unique = torch.unique(
            edge_label_index_concat.t(), dim=0).t()

        batch[edge_type].edge_index = edge_index_message_passing_concat
        reverse_edge_type = (DST_ENTITY_TABLE, DST_ENTITY_COL,
                             SRC_ENTITY_TABLE)
        batch[reverse_edge_type].edge_index = (
            edge_index_message_passing_concat.flip(dims=[0]))

        # Optimization
        optimizer.zero_grad()
        if args.model == "idgnn":
            out = model(batch, SRC_ENTITY_TABLE, DST_ENTITY_TABLE).flatten()
            batch_size = batch[SRC_ENTITY_TABLE].batch_size

            # Get target label
            target = torch.isin(
                batch[DST_ENTITY_TABLE].batch +
                batch_size * batch[DST_ENTITY_TABLE].n_id,
                src_batch + batch_size * dst_index,
            ).float()

            loss = F.binary_cross_entropy_with_logits(out, target)
            numel = out.numel()
        elif args.model in ['hybridgnn', 'shallowrhsgnn']:
            logits = model(batch, SRC_ENTITY_TABLE, DST_ENTITY_TABLE)
            loss = sparse_cross_entropy(logits, edge_label_index_concat_unique)
            numel = len(batch[DST_ENTITY_TABLE].batch)

        loss.backward()
        optimizer.step()
        loss_accum += float(loss) * numel
        count_accum += numel
        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    if count_accum == 0:
        warnings.warn(f"Did not sample a single '{DST_ENTITY_TABLE}' "
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
        batch_size = batch[SRC_ENTITY_TABLE].batch_size

        if args.model == "idgnn":
            out = (model.forward(batch, SRC_ENTITY_TABLE,
                                 DST_ENTITY_TABLE).detach().flatten())
            scores = torch.zeros(batch_size, NUM_DST_NODES, device=out.device)
            scores[batch[DST_ENTITY_TABLE].batch,
                   batch[DST_ENTITY_TABLE].n_id] = torch.sigmoid(out)
        elif args.model in ['hybridgnn', 'shallowrhsgnn']:
            out = model(batch, SRC_ENTITY_TABLE, DST_ENTITY_TABLE).detach()
            scores = torch.sigmoid(out)
        else:
            raise ValueError(f"Unsupported model type: {args.model}.")

        _, pred_mini = torch.topk(scores, k=EVAL_K, dim=1)
        pred_list.append(pred_mini)
    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    return pred


def evaluate(
    pred: NDArray,
    target_table: pd.DataFrame,
) -> Dict[str, float]:
    expected_pred_shape = (len(target_table), EVAL_K)
    if pred.shape != expected_pred_shape:
        raise ValueError(
            f"The shape of pred must be {expected_pred_shape}, but "
            f"{pred.shape} given.")

    pred_isin_list = []
    dst_count_list = []
    for true_dst_nodes, pred_dst_nodes in zip(
            target_table[DST_ENTITY_COL],
            pred,
    ):
        pred_isin_list.append(
            np.isin(np.array(pred_dst_nodes), np.array(true_dst_nodes)))
        dst_count_list.append(len(true_dst_nodes))
    pred_isin = np.stack(pred_isin_list)
    dst_count = np.array(dst_count_list)

    return {fn.__name__: fn(pred_isin, dst_count) for fn in METRICS}


for epoch in range(1, args.epochs + 1):
    train_loss = train()
    print(f"{epoch=}, {train_loss=}")

test_pred = test(loader_dict["test"], desc="Test")
test_metrics = evaluate(test_pred, test_df)
print(f"Best test metrics: {test_metrics}")
