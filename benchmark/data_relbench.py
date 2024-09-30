# official implementation
# https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
# where they split data into train/val/test by splitting the LHS ids
# into 116,677/10,000/10,000.
# 'rel-trial': condition-sponsor-run, site-sponsor-run
# 'rel-amazon': user-item-rate, user-item-purchase, user-item-review
# 'rel-stack': user-post-comment, user-post-related
# 'rel-avito': user-ad-visit
# 'rel-hm': user-item-purchase
from __future__ import annotations

import argparse
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from relbench.base import RecommendationTask
from relbench.tasks import get_task
from torch import Tensor
from torch_frame.data.multi_tensor import _batched_arange
from torch_geometric.seed import seed_everything
from torch_geometric.utils import coalesce
from tqdm import tqdm

total_optimization_steps = 0
LINK_PREDICTION_METRIC = "link_prediction_map"


class MultiVAE(torch.nn.Module):
    def __init__(
        self,
        p_dims: list[int],
        q_dims: list[int] | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if q_dims and q_dims[0] != p_dims[-1]:
            raise ValueError("In and Out dimensions must equal to each other.")
        if q_dims and q_dims[-1] != p_dims[0]:
            raise ValueError(
                "Latent dimension for p- and q- network mismatches.")

        self.p_dims = p_dims
        self.q_dims = q_dims or p_dims[::-1]
        self.dropout = dropout
        self.p_layers = torch.nn.ModuleList([
            torch.nn.Linear(d_in, d_out)
            for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
        ])
        self.q_layers = torch.nn.ModuleList([
            torch.nn.Linear(d_in, d_out) for d_in, d_out in zip(
                self.q_dims[:-1],
                # NOTE: Double the last dim of the encoder for mean and logvar,
                # i.e., [q0, ..., qn] -> [q0, ..., qn*2].
                self.q_dims[1:-1] + [self.q_dims[-1] * 2],
            )
        ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.trunc_normal_(layer.bias, std=0.001)

    def forward(self, x: Tensor) -> Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns mean and logvar of q(z|x)."""
        h = F.normalize(x)  # x: [batch_size, num_rhs]
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
        mu, logvar = h[:, :self.q_dims[-1]], h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Returns z sampled from q(z|x) or the mean of q(z|x)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  # sample more if necessary
            return mu + eps * std
        else:
            # Use mean for inference
            return mu

    def decode(self, z: Tensor) -> Tensor:
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h


def loss_function(
    recon_x: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    anneal: float = 1.0,
) -> Tensor:
    recon_loss = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, dim=-1))
    kl = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + anneal * kl


def train(
    model: MultiVAE,
    optimizer: torch.optim.Optimizer,
    train_data: tuple[Tensor, Tensor],
    device: torch.device,
    args,
    epoch: int,
    task: RecommendationTask,
) -> float:
    rowptr, col = train_data
    model.train()
    global total_optimization_steps
    N = len(rowptr) - 1

    idxlist = list(range(N))
    np.random.shuffle(idxlist)
    for start in tqdm(range(0, N, args.batch_size), desc=f"Epoch {epoch:3d}"):
        end = min(start + args.batch_size, N)
        batch_size = end - start
        lhs_index = torch.tensor(idxlist[start:end], dtype=torch.int64,
                                 device=device)
        # count = rowptr[lhs_index + 1] - rowptr[lhs_index]
        # src_batch, arange = _batched_arange(count)
        # dst_index = col[arange + rowptr[lhs_index][src_batch]]
        src_batch, dst_index = get_rhs_index(lhs_index, rowptr, col)
        # convert rowptr and col to a dense tensor of ones:
        x = torch.zeros(
            (batch_size, task.num_dst_nodes),
            dtype=torch.float32,
            device=device,
        )
        x[src_batch, dst_index] = 1.0
        if args.total_anneal_steps > 0:
            anneal = min(
                args.anneal_cap,
                total_optimization_steps / args.total_anneal_steps,
            )
        else:
            anneal = args.anneal_cap
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar, anneal)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_optimization_steps += 1
    optimizer.zero_grad()
    return loss.item()


def get_rhs_index(lhs_index: Tensor, rowptr: Tensor, col: Tensor) -> Tensor:
    src_batch, arange = _batched_arange(rowptr[lhs_index + 1] -
                                        rowptr[lhs_index])
    dst_index = col[arange + rowptr[lhs_index][src_batch]]
    return src_batch, dst_index


@torch.no_grad()
def test(
    model: MultiVAE,
    data_dict: tuple[Tensor, Tensor],
    device: torch.device,
    args,
    epoch: int,
    task: RecommendationTask,
    stage: Literal["val", "test"],
) -> float:
    model.eval()
    rowptr, col = data_dict[stage]
    N = len(rowptr) - 1
    pred_list: list[Tensor] = []
    for start in tqdm(range(0, N, args.batch_size), desc=f"Epoch {epoch:3d}"):
        end = min(start + args.batch_size, N)
        batch_size = end - start
        lhs_index = torch.tensor(idxlist[start:end], dtype=torch.int64,
                                 device=device)
        src_batch, dst_index = get_rhs_index(lhs_index, rowptr, col)
        # convert rowptr and col to a dense tensor of ones:
        # x_input is from the training set for the validation set,
        # and from the training+validation set for the test set:
        x_input = torch.zeros(
            (batch_size, task.num_dst_nodes),
            dtype=torch.float32,
            device=device,
        )
        x_input[src_batch, dst_index] = 1.0
        recon_x, _, _ = model(x_input)
        scores = torch.sigmoid(recon_x)
        _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
        pred_list.append(pred_mini)

    pred = torch.cat(pred_list, dim=0).cpu().numpy()
    res = task.evaluate(pred, task.get_table(stage))
    return res[LINK_PREDICTION_METRIC]


def load_data_dict(
    task: RecommendationTask,
    device: torch.device,
) -> dict[str, tuple[Tensor, Tensor]]:
    data_dict: dict[str, tuple[Tensor, Tensor]] = {}
    for split in ['train', 'val', 'test']:
        split_df = task.get_table(split).df.drop(
            columns=['timestamp']).explode(task.dst_entity_col)
        edge_index = torch.tensor(
            [
                split_df[task.src_entity_col].values,
                split_df[task.dst_entity_col].values,
            ],
            dtype=torch.int64,
            device=device,
        )
        row, col = coalesce(edge_index)
        rowptr = torch._convert_indices_from_coo_to_csr(
            input=row,
            size=task.num_src_nodes,
        )
        data_dict[split] = (rowptr, col)
    return data_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='rel-trial',
        choices=[
            'rel-trial', 'rel-amazon', 'rel-stack', 'rel-avito', 'rel-hm'
        ],
    )
    parser.add_argument('--task', type=str, default='site-sponsor-run')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.00)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--total_anneal_steps', type=int, default=200_000)
    parser.add_argument('--anneal_cap', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--result_path', type=str, default='result.pt')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task: RecommendationTask = get_task(args.dataset, args.task, download=True)
    data_dict = load_data_dict(task, device)
    seed_everything(args.seed)
    p_dims = [200, 600, task.num_dst_nodes]
    model = MultiVAE(p_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 weight_decay=args.wd)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model,
            optimizer,
            data_dict["train"],
            device,
            args,
            epoch,
            task,
        )
        # val_map = test(
        #     model,
        #     data_dict["val"],
        #     device,
        #     args,
        #     epoch,
        #     task,
        #     "val",
        # )
        print(f'Epoch {epoch:3d}, '
              f'train_loss {train_loss:4.2f}, '
              # f'val_map {val_map:4.2f}'
              )

    # TODO: test from saved model
    # with open(args.result_path, 'rb') as f:
    #     model = torch.load(f)


if __name__ == '__main__':
    main()
