from typing import Any, Dict

import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from hybridgnn.nn.encoder import (
    DEFAULT_STYPE_ENCODER_DICT,
    HeteroEncoder,
    HeteroTemporalEncoder,
)
from hybridgnn.nn.models import HeteroGraphSAGE
from hybridgnn.nn.models.transformer import RHSTransformer
from torch_scatter import scatter_max


class Hybrid_RHSTransformer(torch.nn.Module):
    r"""Implementation of RHSTransformer model.
    Args:
        data (HeteroData): dataset
        col_stats_dict (Dict[str, Dict[str, Dict[StatType, Any]]]): column stats
        num_nodes (int): number of nodes,
        num_layers (int): number of mp layers,
        channels (int): input dimension,
        embedding_dim (int): embedding dimension size,
        aggr (str): aggregation type,
        norm (norm): normalization type,
        dropout (float): dropout rate for the transformer float,
        heads (int): number of attention heads,
        pe (str): type of positional encoding for the transformer,"""
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_nodes: int,
        num_layers: int,
        channels: int,
        embedding_dim: int,
        aggr: str = 'sum',
        norm: str = 'layer_norm',
        dropout: float = 0.2,
        heads: int = 1,
        pe: str = "abs",
    ) -> None:
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            stype_encoder_cls_kwargs=DEFAULT_STYPE_ENCODER_DICT,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=1,
            norm=norm,
            num_layers=1,
        )
        self.lhs_projector = torch.nn.Linear(channels, embedding_dim)
        self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.rhs_embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.lin_offset_idgnn = torch.nn.Linear(embedding_dim, 1)
        self.lin_offset_embgnn = torch.nn.Linear(embedding_dim, 1)

        self.rhs_transformer = RHSTransformer(in_channels=channels,
                                              out_channels=channels,
                                              hidden_channels=channels,
                                              heads=heads, dropout=dropout,
                                              position_encoding=pe)

        self.channels = channels

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.id_awareness_emb.reset_parameters()
        self.rhs_embedding.reset_parameters()
        self.lin_offset_embgnn.reset_parameters()
        self.lin_offset_idgnn.reset_parameters()
        self.lhs_projector.reset_parameters()
        self.rhs_transformer.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
        dst_entity_col: NodeType,
    ) -> Tensor:
        # print ("time dict has the following keys")
        # print (batch.time_dict.keys())
        # dict_keys(['drop_withdrawals', 'outcomes', 'outcome_analyses', 'eligibilities', 'sponsors_studies', 'facilities_studies', 'interventions_studies', 'studies', 'designs', 'reported_event_totals', 'conditions_studies'])

        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        # Add ID-awareness to the root node
        x_dict[entity_table][:seed_time.size(0
                                             )] += self.id_awareness_emb.weight
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                              batch.batch_dict)

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        batch_size = seed_time.size(0)
        lhs_embedding = x_dict[entity_table][:
                                             batch_size]  # batch_size, channel
        lhs_embedding_projected = self.lhs_projector(lhs_embedding)
        rhs_gnn_embedding = x_dict[dst_table]  # num_sampled_rhs, channel
        rhs_idgnn_index = batch.n_id_dict[dst_table]  # num_sampled_rhs
        lhs_idgnn_batch = batch.batch_dict[dst_table]  # batch_size

        #! need custom code to work for specific datasets
        # rhs_time = self.get_rhs_time_dict(batch.time_dict, batch.edge_index_dict, batch[entity_table].seed_time, batch, dst_entity_col, dst_table)

        # adding rhs transformer
        rhs_gnn_embedding = self.rhs_transformer(rhs_gnn_embedding,
                                                 lhs_idgnn_batch, batch_size=batch_size)

        rhs_embedding = self.rhs_embedding  # num_rhs_nodes, channel
        embgnn_logits = lhs_embedding_projected @ rhs_embedding.weight.t(
        )  # batch_size, num_rhs_nodes

        # Model the importance of embedding-GNN prediction for each lhs node
        embgnn_offset_logits = self.lin_offset_embgnn(
            lhs_embedding_projected).flatten()
        embgnn_logits += embgnn_offset_logits.view(-1, 1)

        # Calculate idgnn logits
        idgnn_logits = self.head(
            rhs_gnn_embedding).flatten()  # num_sampled_rhs
        # Because we are only doing 2 hop, we are not really sampling info from
        # lhs therefore, we need to incorporate this information using
        # lhs_embedding[lhs_idgnn_batch] * rhs_gnn_embedding
        idgnn_logits += (
            lhs_embedding[lhs_idgnn_batch] *  # num_sampled_rhs, channel
            rhs_gnn_embedding).sum(
                dim=-1).flatten()  # num_sampled_rhs, channel

        # Model the importance of ID-GNN prediction for each lhs node
        idgnn_offset_logits = self.lin_offset_idgnn(
            lhs_embedding_projected).flatten()
        idgnn_logits = idgnn_logits + idgnn_offset_logits[lhs_idgnn_batch]

        embgnn_logits[lhs_idgnn_batch, rhs_idgnn_index] = idgnn_logits
        return embgnn_logits

    def get_rhs_time_dict(
        self,
        time_dict,
        edge_index_dict,
        seed_time,
        batch_dict,
        dst_entity_col,
        dst_entity_table,
    ):
        # edge_index_dict keys
        """
        dict_keys([('drop_withdrawals', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'drop_withdrawals'), 
        ('outcomes', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'outcomes'), 
        ('outcome_analyses', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'outcome_analyses'), 
        ('outcome_analyses', 'f2p_outcome_id', 'outcomes'), 
        ('outcomes', 'rev_f2p_outcome_id', 'outcome_analyses'), 
        ('eligibilities', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'eligibilities'), 
        ('sponsors_studies', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'sponsors_studies'), 
        ('sponsors_studies', 'f2p_sponsor_id', 'sponsors'), 
        ('sponsors', 'rev_f2p_sponsor_id', 'sponsors_studies'), 
        ('facilities_studies', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'facilities_studies'), 
        ('facilities_studies', 'f2p_facility_id', 'facilities'), 
        ('facilities', 'rev_f2p_facility_id', 'facilities_studies'), 
        ('interventions_studies', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'interventions_studies'), 
        ('interventions_studies', 'f2p_intervention_id', 'interventions'), 
        ('interventions', 'rev_f2p_intervention_id', 'interventions_studies'), 
        ('designs', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'designs'), 
        ('reported_event_totals', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'reported_event_totals'), 
        ('conditions_studies', 'f2p_nct_id', 'studies'), 
        ('studies', 'rev_f2p_nct_id', 'conditions_studies'), 
        ('conditions_studies', 'f2p_condition_id', 'conditions'), 
        ('conditions', 'rev_f2p_condition_id', 'conditions_studies')])
        """
        #* what to put when transaction table is merged
        edge_index = edge_index_dict['sponsors','f2p_sponsor_id',
                                     'sponsors_studies']
        rhs_time, _ = scatter_max(
            time_dict['sponsors'][edge_index[0]],
            edge_index[1])
        SECONDS_IN_A_DAY = 60 * 60 * 24
        NANOSECONDS_IN_A_DAY = 60 * 60 * 24 * 1000000000
        rhs_rel_time = seed_time[batch_dict[dst_entity_col]] - rhs_time
        rhs_rel_time = rhs_rel_time / NANOSECONDS_IN_A_DAY
        return rhs_rel_time
