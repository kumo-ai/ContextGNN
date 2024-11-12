from typing import List, Optional

import numpy as np
import torch


def calculate_hit_rate_ndcg(pred: torch.Tensor, target: List[Optional[int]],
                            top_k: Optional[int] = None):
    r"""Calculates hit rate when pred is a tensor and target is a list.

    Args:
        pred (torch.Tensor): Prediction tensor of size (num_entity,
            num_target_predicitons_per_entity).
        target (List[int]): A list of shape num_entity, where the
            value is None if user doesn't have a next best action.
            The value is the dst node id if there is a next best
            action.
        top_k(int, optional): top_k metrics to look at
    Returns:
        (float, float): hit rate and ndcg score
    """
    hits = 0
    total = 0
    ndcg = 0
    k = top_k if top_k is not None else len(pred[0])
    for i in range(len(target)):
        if target[i] is not None:
            total += 1
            if target[i] in pred[i][:k]:
                hits += 1
                if k != 1:
                    ndcg += np.reciprocal(
                        np.log2(pred[i][:k].tolist().index(target[i]) + 2))

    return hits / total, ndcg / total


def calculate_hit_rate_on_sparse_target(pred: torch.Tensor,
                                        target: torch.sparse.Tensor):
    r"""Calculates hit rate when pred is a tensor and target is a sparse
    tensor
    Args:
        pred (torch.Tensor): Prediction tensor of size (num_entity,
            num_target_predicitons_per_entity).
        target (torch.sparse.Tensor): Target sparse tensor.
    """
    crow_indices = target.crow_indices().to(pred.device)
    col_indices = target.col_indices().to(pred.device)
    values = target.values()
    assert values is not None
    # Iterate through each row and check if predictions match ground truth
    hits = 0
    num_rows = pred.shape[0]

    for i in range(num_rows):
        # Get the ground truth indices for this row
        row_start = crow_indices[i].item()
        row_end = crow_indices[i + 1].item()
        assert isinstance(row_start, int)
        assert isinstance(row_end, int)
        dst_indices = col_indices[row_start:row_end]
        bool_indices = values[row_start:row_end]
        true_indices = dst_indices[bool_indices]

        # Check if any of the predicted values match the true indices
        pred_indices = pred[i]
        if torch.isin(pred_indices, true_indices).any():
            hits += 1

    # Callculate hit rate
    hit_rate = hits / num_rows
    return hit_rate
