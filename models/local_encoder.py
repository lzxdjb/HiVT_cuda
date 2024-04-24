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
from lib2to3.pgen2.token import RPAR
from typing import Optional, Tuple, List, Dict
from torch.utils.cpp_extension import load

import graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from mypyg.data import Data, combine_batch_data
from mypyg.conv import MessagePassing
from mypyg.utils import softmax, subgraph

from models import MultipleInputEmbedding
from models import SingleInputEmbedding
from utils import distance_drop_edge, init_weights
from mypyg.utils import scatter


class LocalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel
        self.local_radius = float(local_radius)
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel)
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)

        self.al_encoder = ALEncoder(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        for t in range(self.historical_steps):
            data[f'edge_index_{t}'], _ = subgraph(
                subset=~data['padding_mask'][:, t], edge_index=data['edge_index'])
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - \
                data['positions'][data[f'edge_index_{t}'][1], t]

            data[f'edge_ati_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][1], t] - \
                data['positions'][data[f'edge_index_{t}'][0], t] 
                
        # if self.parallel:
        # print("x",data['x'].shape)
        snapshots: List[Data] = []
        ati_snapshots: List[Data] = []
        for t in range(self.historical_steps):

            edge_index, edge_attr = distance_drop_edge(
                self.local_radius, data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
            edge_ati_index, edge_ati_attr = distance_drop_edge(
                self.local_radius, data[f'edge_index_{t}'], data[f'edge_ati_attr_{t}'])

            snapshots.append(Data(num_nodes=data.num_nodes, kwargs={
                             'x': data['x'][:, t], 'edge_index': edge_index, 'edge_attr': edge_attr}))
            ati_snapshots.append(Data(num_nodes=data.num_nodes, kwargs={
                             'x': data['x'][:, t], 'edge_index': edge_ati_index, 'edge_attr': edge_ati_attr}))

        batch = combine_batch_data(snapshots)
        ati_batch = combine_batch_data(ati_snapshots)
        # print("batch",batch['x'].shape)
        out = self.aa_encoder(x=batch['x'], t=None, edge_index=batch['edge_index'], edge_attr=batch['edge_attr'],
                              bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'] , edge_ati_attr = ati_batch['edge_attr'])

        out = out.view(self.historical_steps,
                       out.shape[0] // self.historical_steps, -1)
       

        # print("padding_mask",data['padding_mask'][:, : self.historical_steps].shape)
        out = self.temporal_encoder(
            x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
     

      

        edge_index, edge_attr = distance_drop_edge(
            self.local_radius, data['lane_actor_index'], data['lane_actor_vectors'])


        out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
                              is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],
                              traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat'])
        return out

# graph_cu = load(name="graph", sources=["ops/graph.cu"] , extra_include_paths = "/home/lzx/anaconda3/lib/python3.11/site-packages")
file_cnt = 0

class AAEncoder(MessagePassing):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 **kwargs) -> None:
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        self.center_embed = SingleInputEmbedding(
            in_channel=node_dim, out_channel=embed_dim)
        # print(node_dim)
        # print(embed_dim)
        self.nbr_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        # print("embed_dim",embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
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
        self.bos_token = nn.Parameter(
            torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                t: Optional[int],
                edge_ati_attr: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                bos_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        global file_cnt
        if 1:
            output, counts = torch.unique_consecutive(edge_index[0], return_counts=True)
            torch.save(counts, f"cnts/count{file_cnt}.pt")
            file_cnt += 1
            counts = torch.cumsum(counts, dim = 0)
    
            edge_index_1 = edge_index[1]
            edge_index_0 = edge_index[0]
          
           

            #######  test
            test_input = torch.randn([1,2]).cuda()
            test_weight_linear = torch.ones([64,2]).cuda()
            test_bias_linear = torch.zeros([64]).cuda()
            mat_1 = torch.randn([1,2]).cuda()
            #######
            
            # test_center_embed = graph.graph_max(x, edge_index_1,edge_index_0,edge_attr,edge_ati_attr , bos_mask.t(),rotate_mat, mat_1 , 
                 
            #                          self.center_embed.embed[0].weight,
            #                          self.center_embed.embed[0].bias,
            #                          self.center_embed.embed[3].weight,
            #                          self.center_embed.embed[3].bias,
            #                          self.center_embed.embed[6].weight,
            #                          self.center_embed.embed[6].bias,
                                     
            #                          self.center_embed.embed[1].weight,
            #                          self.center_embed.embed[1].bias,
            #                          self.center_embed.embed[4].weight,
            #                          self.center_embed.embed[4].bias,
            #                          self.center_embed.embed[7].weight,
            #                          self.center_embed.embed[7].bias,
                                     
            #                          self.bos_token,
                                    
            #                          self.norm1.weight,
            #                          self.norm1.bias,

            #                          output,
            #                          counts,

            #                         self.nbr_embed.module_list[0][0].weight,
            #                         self.nbr_embed.module_list[0][0].bias,
            #                         self.nbr_embed.module_list[0][1].weight,
            #                         self.nbr_embed.module_list[0][1].bias,
            #                         self.nbr_embed.module_list[0][3].weight,
            #                         self.nbr_embed.module_list[0][3].bias,

            #                         self.nbr_embed.module_list[1][0].weight,
            #                         self.nbr_embed.module_list[1][0].bias,
            #                         self.nbr_embed.module_list[1][1].weight,
            #                         self.nbr_embed.module_list[1][1].bias,
            #                         self.nbr_embed.module_list[1][3].weight,
            #                         self.nbr_embed.module_list[1][3].bias,

            #                         self.nbr_embed.aggr_embed[0].weight,
            #                         self.nbr_embed.aggr_embed[0].bias,
            #                         self.nbr_embed.aggr_embed[2].weight,
            #                         self.nbr_embed.aggr_embed[2].bias,
            #                         self.nbr_embed.aggr_embed[3].weight,
            #                         self.nbr_embed.aggr_embed[3].bias,

            #                         self.lin_q.weight,
            #                         self.lin_q.bias,
            #                         self.lin_k.weight,
            #                         self.lin_k.bias,
            #                         self.lin_v.weight,
            #                         self.lin_v.bias,

            #                         self.lin_ih.weight,
            #                         self.lin_ih.bias,
            #                         self.lin_hh.weight,
            #                         self.lin_hh.bias,
            #                         self.lin_self.weight,
            #                         self.lin_self.bias,
            #                         self.out_proj.weight,
            #                         self.out_proj.bias,

            #                         self.norm2.weight,
            #                         self.norm2.bias,

            #                         self.mlp[0].weight,
            #                         self.mlp[0].bias,
            #                         self.mlp[3].weight,
            #                         self.mlp[3].bias,
             
            #                         test_weight_linear,
            #                         test_bias_linear,
            #                         test_input                           
            #                         ) 
            
            # test_center_embed = graph_cu.graph_max(x, edge_index_1,edge_index_0,edge_attr,edge_ati_attr , bos_mask.t(),rotate_mat, mat_1 , 
                 
            #                          self.center_embed.embed[0].weight,
            #                          self.center_embed.embed[0].bias,
            #                          self.center_embed.embed[3].weight,
            #                          self.center_embed.embed[3].bias,
            #                          self.center_embed.embed[6].weight,
            #                          self.center_embed.embed[6].bias,
                                     
            #                          self.center_embed.embed[1].weight,
            #                          self.center_embed.embed[1].bias,
            #                          self.center_embed.embed[4].weight,
            #                          self.center_embed.embed[4].bias,
            #                          self.center_embed.embed[7].weight,
            #                          self.center_embed.embed[7].bias,
                                     
            #                          self.bos_token,
                                    
            #                          self.norm1.weight,
            #                          self.norm1.bias,

            #                          output,
            #                          counts,

            #                         self.nbr_embed.module_list[0][0].weight,
            #                         self.nbr_embed.module_list[0][0].bias,
            #                         self.nbr_embed.module_list[0][1].weight,
            #                         self.nbr_embed.module_list[0][1].bias,
            #                         self.nbr_embed.module_list[0][3].weight,
            #                         self.nbr_embed.module_list[0][3].bias,

            #                         self.nbr_embed.module_list[1][0].weight,
            #                         self.nbr_embed.module_list[1][0].bias,
            #                         self.nbr_embed.module_list[1][1].weight,
            #                         self.nbr_embed.module_list[1][1].bias,
            #                         self.nbr_embed.module_list[1][3].weight,
            #                         self.nbr_embed.module_list[1][3].bias,

            #                         self.nbr_embed.aggr_embed[0].weight,
            #                         self.nbr_embed.aggr_embed[0].bias,
            #                         self.nbr_embed.aggr_embed[2].weight,
            #                         self.nbr_embed.aggr_embed[2].bias,
            #                         self.nbr_embed.aggr_embed[3].weight,
            #                         self.nbr_embed.aggr_embed[3].bias,

            #                         self.lin_q.weight,
            #                         self.lin_q.bias,
            #                         self.lin_k.weight,
            #                         self.lin_k.bias,
            #                         self.lin_v.weight,
            #                         self.lin_v.bias,

            #                         self.lin_ih.weight,
            #                         self.lin_ih.bias,
            #                         self.lin_hh.weight,
            #                         self.lin_hh.bias,
            #                         self.lin_self.weight,
            #                         self.lin_self.bias,
            #                         self.out_proj.weight,
            #                         self.out_proj.bias,

            #                         self.norm2.weight,
            #                         self.norm2.bias,

            #                         self.mlp[0].weight,
            #                         self.mlp[0].bias,
            #                         self.mlp[3].weight,
            #                         self.mlp[3].bias,
             
            #                         test_weight_linear,
            #                         test_bias_linear,
            #                         test_input                           
            #                         ) 
            # print("test_center_embed",test_center_embed)


      
            x_temp = x.view(self.historical_steps,
                            x.shape[0] // self.historical_steps, -1).unsqueeze(-2)
            rotate_mat_temp1 = rotate_mat.expand(
                self.historical_steps, rotate_mat.shape[0], 2, 2)
            mat_1 = torch.matmul(x_temp, rotate_mat_temp1)

            center_embed = self.center_embed(mat_1)

            center_embed = center_embed.view(self.historical_steps,
                            x.shape[0] // self.historical_steps, -1)
                            

            center_embed = torch.where(bos_mask.t().unsqueeze(-1),
                                   self.bos_token.unsqueeze(-2),
                                   center_embed).reshape(x.shape[0], -1)
            center_embed = self.norm1(center_embed)


        
        center_embed = center_embed + \
            self._mha_block(center_embed, x,
                            edge_index, edge_attr, rotate_mat , center_embed)
        
        # print("center_embed" , center_embed)
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))


        # print("center_embed",center_embed)
        return center_embed

    def message(self,
                edge_index: torch.Tensor,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                size_i: int) -> torch.Tensor:

        if 1:


            center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[
                edge_index[1]]  
            bmm1 = torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2)

            bmm2 = torch.bmm(edge_attr.unsqueeze(-2),
                             center_rotate_mat).squeeze(-2)
            # print("bmm2",bmm2)

            nbr_embed = self.nbr_embed([bmm1, bmm2])
            # print("nbr_embed",nbr_embed)
            ###test

            # self.nbr_embed.module_list[0][0].weight,
            # self.nbr_embed.module_list[0][0].bias,
            # self.nbr_embed.module_list[0][1].weight,
            # self.nbr_embed.module_list[0][1].bias,
            # self.nbr_embed.module_list[0][3].weight,
            # self.nbr_embed.module_list[0][3].bias,
            
            # self.nbr_embed.module_list[1][0].weight,
            # self.nbr_embed.module_list[1][0].bias,
            # self.nbr_embed.module_list[1][1].weight,
            # self.nbr_embed.module_list[1][1].bias,
            # self.nbr_embed.module_list[1][3].weight,
            # self.nbr_embed.module_list[1][3].bias,
            # center_embed_1 = F.linear(bmm2,self.nbr_embed.module_list[1][0].weight , self.nbr_embed.module_list[1][0].bias)
            # center_embed_1 = F.layer_norm(center_embed_1, [64], self.nbr_embed.module_list[1][1].weight, self.nbr_embed.module_list[1][1].bias, 1e-5)
            # center_embed_1 = F.relu(center_embed_1 , inplace=True)
            # center_embed_1 = F.linear(center_embed_1,self.nbr_embed.module_list[1][3].weight , self.nbr_embed.module_list[1][3].bias)
            # print("center_embed_1",center_embed_1)

            #####

       
        query = self.lin_q(center_embed_i).view(-1,
                                                self.num_heads, self.embed_dim // self.num_heads)
        # print("query",query.view(-1 , 64))

        key = self.lin_k(nbr_embed).view(-1, self.num_heads,
                                         self.embed_dim // self.num_heads)

        value = self.lin_v(nbr_embed).view(-1, self.num_heads,
                                           self.embed_dim // self.num_heads)
        # print("after_linear",self.lin_v(nbr_embed))

        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
    
        # print("djb",alpha.view( -1, 8))
        # print("index",index.shape)
        # print("size_i",size_i)
        alpha = softmax(alpha, index, size_i)
        # print("before" , alpha)

        alpha = self.attn_drop(alpha)
        # print("after" , alpha)

        # print("alpha",alpha.unsqueeze(-1).shape)
        # print("value" , value.shape)
        # print("alpha.unsqueseze(-1)",alpha.unsqueeze(-1).shape)
        temp2 = value * alpha.unsqueeze(-1)
        # print("temp2",temp2.view(-1 , 64))
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        # print("before input",inputs.shape)
        inputs = inputs.view(-1, self.embed_dim)
        # print("after input",inputs.shape)

        temp = self.lin_ih(inputs) + self.lin_hh(center_embed)
        # print("temp" , temp)

        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        # print("gate" , inputs + gate * (self.lin_self(center_embed) - inputs))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

    def propagate(self, edge_index: torch.Tensor, x, center_embed, edge_attr, rotate_mat: Optional[torch.Tensor] , test_center_embed):
        size: List[int] = [-1, -1]
        i, j = 1, 0

        # print("center_embed",center_embed.shape)
        # print("index",edge_index)
        # print("index",edge_index.shape)
        center_embed_i = self._collect(center_embed, edge_index, size, i)

        # print("center_embed_i ",center_embed_i.shape)

        x_j = self._collect(x, edge_index, size, j)
        # print("x_j",x_j.shape)
        # print("size",size)

        index = edge_index[i]
        # print("index.shape",index.shape)
        dim_size = size_i = size[i] if size[i] >= 0 else size[j]
        # print("dim_size",dim_size)
        # print("out",center_embed_i.shape)
        out = self.message(edge_index, center_embed_i, x_j,
                           edge_attr, rotate_mat, index, size_i)
        # print("out",out.shape)
        out = self.aggr_module(out, index, dim_size=dim_size,
                               dim=self.node_dim)
        # print("out",out.view(-1 , 64))
        # out = scatter(out,index,dim_size = dim_size, dim=-2)
        # out = self.aggregate(out, index, dim_size)
        # print("out",out.shape)
        # print("center_embed",center_embed.shape)
        out = self.update(out, center_embed)
        # print("out",out.shape)
        return out

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   test_center_embed:torch.Tensor) -> torch.Tensor:

        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, rotate_mat=rotate_mat, test_center_embed = test_center_embed))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # see https://github.com/pytorch/pytorch/pull/102045
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(
            torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)

        # print("x",x.shape)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        # print("self.cls_token",self.cls_token.shape)
        # print("expand_cls_token",expand_cls_token.shape)

        x = torch.cat((x, expand_cls_token), dim=0)
        # print("x",x.shape)
        x = x + self.pos_embed
        # print("pos_embed",self.pos_embed.shape)
        # print("mask",self.attn_mask.shape)
        out = self.transformer_encoder(
            src=x, mask=self.attn_mask, src_key_padding_mask=None)
        # print("out",out.shape)
        # print("out[-1]",out[-1].shape)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # print("embed_dim",embed_dim)
        # print("")
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = None) -> torch.Tensor:
        # print("src",src.shape)
        x = src
        # print("before",x.shape)
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        # print("after_embed",self._sa_block(self.norm1(x), src_mask, src_key_padding_mask).shape)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


class ALEncoder(MessagePassing):

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
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
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
        nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
        nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                is_intersections: torch.Tensor,
                turn_directions: torch.Tensor,
                traffic_controls: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        turn_directions = turn_directions.long()
        traffic_controls = traffic_controls.long()
        x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections,
                                            turn_directions, traffic_controls, rotate_mat)
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(self,
                edge_index: torch.Tensor,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                is_intersections_j,
                turn_directions_j,
                traffic_controls_j,
                rotate_mat: Optional[torch.Tensor],
                index: torch.Tensor,
                size_i: int) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed([x_j, edge_attr],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                                   torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],
                                  [self.is_intersection_embed[is_intersections_j],
                                   self.turn_direction_embed[turn_directions_j],
                                   self.traffic_control_embed[traffic_controls_j]])
        query = self.lin_q(x_i).view(-1, self.num_heads,
                                     self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads,
                                   self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads,
                                     self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def propagate(self, edge_index: torch.Tensor, x: Tuple[torch.Tensor, torch.Tensor], edge_attr,
                  is_intersections, turn_directions, traffic_controls, rotate_mat: Optional[torch.Tensor]):
        size: List[int] = [-1, -1]

        i, j = 1, 0
        x_i = self._collect_tuple(x, edge_index, size, i)
        x_j = self._collect_tuple(x, edge_index, size, j)
        is_intersections_j = self._collect(
            is_intersections, edge_index, size, j)
        turn_directions_j = self._collect(turn_directions, edge_index, size, j)
        traffic_controls_j = self._collect(
            traffic_controls, edge_index, size, j)

        index = edge_index[i]
        dim_size = size_i = size[i] if size[i] >= 0 else size[j]

        out = self.message(edge_index, x_i, x_j, edge_attr, is_intersections_j,
                           turn_directions_j, traffic_controls_j, rotate_mat, index, size_i)
        out = self.aggregate(out, index, dim_size)
        out = self.update(out, x)
        return out

    def _mha_block(self,
                   x_actor: torch.Tensor,
                   x_lane: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: torch.Tensor,
                   is_intersections: torch.Tensor,
                   turn_directions: torch.Tensor,
                   traffic_controls: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor]) -> torch.Tensor:
        x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
                                               is_intersections=is_intersections, turn_directions=turn_directions,
                                               traffic_controls=traffic_controls, rotate_mat=rotate_mat))
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)
