import os.path as osp
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_frame import stype
from torch_frame.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index


class IJCAI_Contest():
    r"""IJCAI Contest Dataset.
    Source: https://github.com/akaxlh/TGT/tree/main/Datasets/ijcai.
    """
    def __init__(self, path):
        col_stats_dict = {}
        dst_nodes = None
        trnLabel = None
        behs = ['click', 'fav', 'cart', 'buy']
        src_entity_table = 'user'
        dst_entity_table = 'item'
        data = HeteroData()
        for i in range(len(behs)):
            behavior = behs[i]
            with open(osp.join(path, 'trn_' + behavior), 'rb') as fs:
                mat = pickle.load(fs)
            if i == 0:
                dataset = Dataset(
                    pd.DataFrame({"__const__": np.ones(mat.shape[0])}),
                    col_to_stype={"__const__": stype.numerical})
                dataset.materialize()
                data['user'].tf = dataset.tensor_frame
                col_stats_dict['user'] = dataset.col_stats
                dataset = Dataset(
                    pd.DataFrame({"__const__": np.ones(mat.shape[1])}),
                    col_to_stype={"__const__": stype.numerical})
                dataset.materialize()
                data['item'].tf = dataset.tensor_frame
                col_stats_dict['item'] = dataset.col_stats
                data['user'].n_id = torch.arange(mat.shape[0],
                                                 dtype=torch.long)
                data['item'].n_id = torch.arange(mat.shape[1],
                                                 dtype=torch.long)
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
                trnLabel = 1 * (mat != 0)
            beh_idx = torch.arange(len(coo_mat.data), dtype=torch.long)
            self.create_edge(data, behavior, beh_idx, src_entity_table,
                             torch.tensor(coo_mat.row, dtype=torch.long))
            self.create_edge(data, behavior, beh_idx, dst_entity_table,
                             torch.tensor(coo_mat.col, dtype=torch.long))
            # HeteroData obj
            self.datat = data
            self.dst_nodes = dst_nodes
            self.trnLabel = trnLabel
            self.col_stats_dict = col_stats_dict

    def create_edge(self, data, behavior, beh_idx, pkey_name, pkey_idx):
        # fkey -> pkey edges
        edge_index = torch.stack([beh_idx, pkey_idx], dim=0)
        edge_type = (behavior, f"f2p_{behavior}", pkey_name)
        data[edge_type].edge_index = sort_edge_index(edge_index)

        # pkey -> fkey edges.
        # "rev_" is added so that PyG loader recognizes the reverse edges
        edge_index = torch.stack([pkey_idx, beh_idx], dim=0)
        edge_type = (pkey_name, f"rev_f2p_{behavior}", behavior)
        data[edge_type].edge_index = sort_edge_index(edge_index)
