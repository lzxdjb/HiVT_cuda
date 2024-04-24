# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from mypyg.data import Data
from mypyg.conv import MessagePassing
from mypyg.utils import softmax, subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import init_weights


class GlobalInteractor(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 edge_dim: int,
                 num_modes: int = 6,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 rotate: bool = True) -> None:
        super(GlobalInteractor, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes

        if rotate:
            self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        else:
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        self.global_interactor_layers = nn.ModuleList(
            [GlobalInteractorLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.apply(init_weights)

    def forward(self,
                data: Data,
                local_embed: torch.Tensor) -> torch.Tensor:
        edge_index, _ = subgraph(subset=~data['padding_mask'][:, self.historical_steps - 1], edge_index=data['edge_index'])
        rel_pos = data['positions'][edge_index[0], self.historical_steps - 1] - data['positions'][
            edge_index[1], self.historical_steps - 1]
        if data['rotate_mat'] is None:
            rel_embed = self.rel_embed(rel_pos)
        else:
            rel_pos = torch.bmm(rel_pos.unsqueeze(-2), data['rotate_mat'][edge_index[1]]).squeeze(-2)
            rel_theta = data['rotate_angles'][edge_index[0]] - data['rotate_angles'][edge_index[1]]
            rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
            rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
            rel_embed = self.rel_embed([rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)])
        x = local_embed
        for layer in self.global_interactor_layers:
            x = layer(x, edge_index, rel_embed)
        x = self.norm(x)  # [N, D]
        x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)  # [N, F, D]
        x = x.transpose(0, 1)  # [F, N, D]
        return x


class GlobalInteractorLayer(MessagePassing):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(GlobalInteractorLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr)
        x = x + self._ff_block(self.norm2(x))
        return x

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                size_i: int) -> torch.Tensor:
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node + value_edge) * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)

    def propagate(self, edge_index: torch.Tensor, x, edge_attr):
        size: List[int] = [-1, -1]

        i, j = 1, 0
        x_i = self._collect(x, edge_index, size, i)
        x_j = self._collect(x, edge_index, size, j)
        
        index = edge_index[i]
        dim_size = size_i = size[i] if size[i] >= 0 else size[j]

        out = self.message(x_i, x_j, edge_attr, index, size_i)
        out = self.aggregate(out, index, dim_size)
        out = self.update(out, x)
        return out

    def _mha_block(self,
                   x: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.out_proj(self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr))
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
