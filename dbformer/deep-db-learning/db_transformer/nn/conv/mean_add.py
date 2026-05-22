from typing import List, Optional, Union

import torch

from torch_geometric.nn import MessagePassing, Aggregation


class MeanAddConv(MessagePassing):


    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        per_column_embedding: bool = True,
    ):
        super().__init__(aggr=aggr, node_dim=-3 if per_column_embedding else -2)

    def forward(self, x, edge_index):


        if isinstance(x, (tuple, list)):
            x_src, x_dst = x[0], x[1]
        else:
            x_src = x_dst = x


        agg = self.propagate(edge_index, x=(x_src, x_dst))


        return x_dst + agg

    def message(self, x_j: torch.Tensor):


        return x_j.mean(dim=1, keepdim=True)