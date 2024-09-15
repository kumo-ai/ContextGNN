"""Example script to run the models in this repository.

python static_example.py
"""

import argparse
import os
import os.path as osp
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
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

from hybridgnn.nn.models import HybridGNN
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
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
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

PSEUDO_TIME = "pseudo_time"
SRC_ENTITY_TABLE = "user_table"
DST_ENTITY_TABLE = "item_table"
SRC_ENTITY_COL = "user_id"
DST_ENTITY_COL = "item_id"
EVAL_K = 12
METRICS = [
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
]

dataset = args.dataset
input_data_dir = osp.join("../", "static_data", dataset)
tune_metric = "link_prediction_map"

# Load user data
user_path = osp.join(input_data_dir, "user_list.txt")
lhs_df = pd.read_csv(user_path, delim_whitespace=True)
# Drop `org_id` and rename `remap_id` to `user_id`
lhs_df = lhs_df.drop(columns=['org_id']).rename(
    columns={'remap_id': SRC_ENTITY_COL})

# Load item data
item_path = osp.join(input_data_dir, "item_list.txt")
rhs_df = pd.read_csv(item_path, delim_whitespace=True)
# Drop `org_id` and rename `remap_id` to `item_id`
rhs_df = rhs_df.drop(columns=['org_id']).rename(
    columns={'remap_id': DST_ENTITY_COL})
num_dst_nodes = len(rhs_df)

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
pseudo_times = pd.date_range(start=pd.Timestamp('1970-01-01'),
                             periods=len(train_df), freq='s')
train_df[PSEUDO_TIME] = pseudo_times

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
pseudo_times = pd.date_range(
    start=train_df[PSEUDO_TIME].max() + pd.DateOffset(1), periods=len(test_df),
    freq='s')
test_df[PSEUDO_TIME] = pseudo_times

train_df_explode = train_df.explode(DST_ENTITY_COL).reset_index(drop=True)
test_df_explode = test_df.explode(DST_ENTITY_COL).reset_index(drop=True)
target_df = pd.concat([train_df_explode, test_df_explode], ignore_index=True)

table_dict = {
    "user_table": lhs_df,
    "item_table": rhs_df,
    "transaction_table": target_df,
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
    for table_name, table in table_dict.items():
        df = table
        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey and fkey
        col_to_stype = {"__const__": stype.numerical}
        fkey_dict = {}
        if table_name == "transaction_table":
            # col_to_stype = {"__const__": stype.numerical}
            fkey_col_to_pkey_table = {
                'user_id': 'user_table',
                'item_id': 'item_table'
            }
            fkey_dict = {key: df[key] for key in fkey_col_to_pkey_table}
        df = pd.DataFrame({"__const__": np.ones(len(table)), **fkey_dict})

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
        ).materialize()

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute:
        if PSEUDO_TIME in table.columns:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table[PSEUDO_TIME]))

        if table_name == "transaction_table":
            fkey_col_to_pkey_table = {
                'user_id': 'user_table',
                'item_id': 'item_table'
            }
            for fkey_name, pkey_table_name in fkey_col_to_pkey_table.items():
                pkey_index = df[fkey_name]

                # Filter out dangling foreign keys
                mask = ~pkey_index.isna()
                fkey_index = torch.arange(len(pkey_index))

                # Filter dangling foreign keys:
                pkey_index = torch.from_numpy(
                    pkey_index[mask].astype(int).values)
                fkey_index = fkey_index[torch.from_numpy(mask.values)]

                # Ensure no dangling fkeys
                assert (pkey_index < len(table_dict[pkey_table_name])).all()

                # fkey -> pkey edges
                edge_index = torch.stack([fkey_index, pkey_index], dim=0)
                edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
                data[edge_type].edge_index = sort_edge_index(edge_index)

                # pkey -> fkey edges.
                # "rev_" is added so that PyG loader recognizes the reverse
                # edges
                edge_index = torch.stack([pkey_index, fkey_index], dim=0)
                edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}",
                             table_name)
                data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()
    return data, col_stats_dict


data, col_stats_dict = static_data_make_pkey_fkey_graph(
    table_dict=table_dict,
    col_to_stype_dict=col_to_stype_dict,
)

num_neighbors = [
    int(args.num_neighbors // 2**i) for i in range(args.num_layers)
]


def static_get_link_train_table_input(
    transaction_df: pd.DataFrame,
    num_dst_nodes: int,
) -> LinkTrainTableInput:
    df = transaction_df
    src_entity_col = "user_id"
    dst_entity_col = "item_id"
    src_node_idx: Tensor = torch.from_numpy(
        df[src_entity_col].astype(int).values)
    exploded = df[dst_entity_col].explode()
    coo_indices = torch.from_numpy(
        np.stack([exploded.index.values,
                  exploded.values.astype(int)]))
    sparse_coo = torch.sparse_coo_tensor(
        coo_indices,
        torch.ones(coo_indices.size(1), dtype=bool),
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
        table, num_dst_nodes=num_dst_nodes)
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
        "num_layers": 4,
    },
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    sparse_tensor = SparseTensor(dst_nodes_dict["train"][1], device=device)
    for batch in tqdm(loader_dict["train"], total=total_steps, desc="Train"):
        batch = batch.to(device)

        # Get ground-truth
        input_id = batch[SRC_ENTITY_TABLE].input_id
        src_batch, dst_index = sparse_tensor[input_id]

        # Optimization
        optimizer.zero_grad()

        logits = model(batch, SRC_ENTITY_TABLE, DST_ENTITY_TABLE)
        edge_label_index = torch.stack([src_batch, dst_index], dim=0)
        loss = sparse_cross_entropy(logits, edge_label_index)
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
        out = model(batch, SRC_ENTITY_TABLE, DST_ENTITY_TABLE).detach()
        scores = torch.sigmoid(out)

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

test_pred = test(loader_dict["test"], desc="Test")
test_metrics = evaluate(test_pred, test_df)
print(f"Best test metrics: {test_metrics}")
