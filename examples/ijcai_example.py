import pickle
import argparse
import os.path as osp
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData
from torch_frame import stype
import torch
from torch_geometric.loader import NeighborLoader
import numpy as np
import scipy.sparse as sp
from torch import Tensor
from torch_frame import stype
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-ijcai")
parser.add_argument("--task", type=str, default="user-item-purchase")
parser.add_argument(
    "--model",
    type=str,
    default="hybridgnn",
    choices=["hybridgnn", "idgnn", "shallowrhsgnn"],
)
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
args = parser.parse_args()
seed_everything(args.seed)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'ijcai-contest')
behs = ['click', 'fav', 'cart', 'buy']


data = HeteroData()

def create_edge(data, behavior, beh_idx, pkey_name, pkey_idx):
    # fkey -> pkey edges
    edge_index = torch.stack([beh_idx, pkey_idx],dim=0)
    edge_type = (behavior, f"f2p_{behavior}", pkey_name)
    data[edge_type].edge_index = sort_edge_index(edge_index)

    # pkey -> fkey edges.
    # "rev_" is added so that PyG loader recognizes the reverse edges
    edge_index = torch.stack([pkey_idx, beh_idx], dim=0)
    edge_type = (pkey_name, f"rev_f2p_{behavior}", behavior)
    data[edge_type].edge_index = sort_edge_index(edge_index)

for i in range(len(behs)):
    behavior = behs[i]
    with open(osp.join(path, 'trn_'+behavior), 'rb') as fs:
        mat = pickle.load(fs)
    if i == 0:
        data['user'].x = torch.tensor(np.ones(mat.shape[0])).view(-1, 1)
        data['item'].x = torch.tensor(np.ones(mat.shape[1])).view(-1, 1)
    col_to_stype = {"time": stype.timestamp}
    data[behavior].x = torch.tensor(mat.data)
    coo_mat = sp.coo_matrix(mat)
    beh_idx = torch.arange(len(coo_mat.data))
    create_edge(data, behavior, beh_idx, 'user', torch.tensor(coo_mat.row))
    create_edge(data, behavior, beh_idx, 'item', torch.tensor(coo_mat.col))

num_neighbors = [
    int(args.num_neighbors // 2**i) for i in range(args.num_layers)
]

loader_dict: Dict[str, NeighborLoader] = {}
dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
num_dst_nodes_dict: Dict[str, int] = {}
for split in ["train", "val", "test"]:

    dst_nodes_dict[split] = table_input.dst_nodes
    num_dst_nodes_dict[split] = table_input.num_dst_nodes
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes='user',
        input_time=table_input.src_time,
        subgraph_type="bidirectional",
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )