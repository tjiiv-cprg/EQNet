# Copyright (c) 2023, Zikang Zhou. All rights reserved.
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
import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from qcnet_55cacb4.layers import AttentionLayer
SharedAttentionLayer = AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle

from collections.abc import Mapping
# from trace_commentor import Commentor, silent


class EQNetDecoder(nn.Module):

    def __init__(self, dataset: str, input_dim: int, hidden_dim: int, output_dim: int,
                 output_head: bool, num_historical_steps: int, num_future_steps: int,
                 num_modes: int, num_recurrent_steps: int, num_t2m_steps: Optional[int],
                 pl2m_radius: float, a2m_radius: float, num_freq_bands: int,
                 num_layers: int, num_heads: int, head_dim: int,
                 dropout: float, detach_m: bool, refine: bool) -> None:
        super(EQNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.refine_enabled = refine
        self.detach_m = detach_m

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.query_emb = nn.Embedding(1, hidden_dim)
        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t,
                                          hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m,
                                           hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m,
                                          hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        if self.refine_enabled:
            self.y_emb = FourierEmbedding(input_dim=output_dim + output_head,
                                        hidden_dim=hidden_dim,
                                        num_freq_bands=num_freq_bands)
            self.traj_emb = nn.GRU(input_size=hidden_dim,
                                hidden_size=hidden_dim,
                                num_layers=1,
                                bias=True,
                                batch_first=False,
                                dropout=0.0,
                                bidirectional=False)
            self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2q_propose_attn_layers = nn.ModuleList([
            SharedAttentionLayer(hidden_dim=hidden_dim,
                           num_heads=num_heads,
                           head_dim=head_dim,
                           dropout=dropout,
                           bipartite=True,
                           has_pos_emb=True) for _ in range(num_layers)
        ])
        self.pl2q_propose_attn_layers = nn.ModuleList([
            SharedAttentionLayer(hidden_dim=hidden_dim,
                           num_heads=num_heads,
                           head_dim=head_dim,
                           dropout=dropout,
                           bipartite=True,
                           has_pos_emb=True) for _ in range(num_layers)
        ])
        self.a2q_propose_attn_layers = nn.ModuleList([
            SharedAttentionLayer(hidden_dim=hidden_dim,
                           num_heads=num_heads,
                           head_dim=head_dim,
                           dropout=dropout,
                           bipartite=True,
                           has_pos_emb=True) for _ in range(num_layers)
        ])
        self.q_out_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_recurrent_steps)])

        self.q2m_propose_attn_layers = nn.ModuleList([
            AttentionLayer(hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            bipartite=True,
                            has_pos_emb=False) for _ in range(num_layers)
        ])
        self.m2m_propose_attn_layers = nn.ModuleList([
            AttentionLayer(hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            bipartite=False,
                            has_pos_emb=False) for _ in range(num_layers)
        ])
        if self.refine_enabled:
            self.q2m_refine_attn_layers = nn.ModuleList([
                AttentionLayer(hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            dropout=dropout,
                            bipartite=True,
                            has_pos_emb=False) for _ in range(num_layers)
            ])

        self.m_propose_out_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_recurrent_steps)])
            
        self.m2q_mlp = MLPLayer(
            input_dim=hidden_dim*num_modes,
            hidden_dim=hidden_dim*num_modes,
            output_dim=hidden_dim
        )
        
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim,
                                           hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim //
                                           num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim,
                                             hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim //
                                             num_recurrent_steps)
        if self.refine_enabled:
            self.m_refine_out_norm = nn.LayerNorm(hidden_dim)
            self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim,
                                            hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
            self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim,
                                                hidden_dim=hidden_dim,
                                                output_dim=num_future_steps * output_dim)

        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim,
                                                hidden_dim=hidden_dim,
                                                output_dim=num_future_steps //
                                                num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim,
                                                 hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps //
                                                 num_recurrent_steps)
            if self.refine_enabled:
                self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim,
                                                hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
                self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim,
                                                    hidden_dim=hidden_dim,
                                                    output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            if self.refine_enabled:
                self.to_loc_refine_head = None
                self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(self, data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        A = data['agent']['num_nodes']
        Pl = data['map_polygon']['num_nodes']
        Pt = data['map_point']['num_nodes']
        M = self.num_modes
        D = self.hidden_dim
        Th = self.num_historical_steps
        Tf = self.num_future_steps
        Trec = self.num_recurrent_steps
        Ci = self.input_dim
        Co = self.output_dim
        dev = data['agent']['position'].device

        pos_q = data['agent']['position'][:, Th - 1, :Ci]  # [A, C]
        head_q = data['agent']['heading'][:, Th - 1]  # [A]
        head_vector_q = torch.stack([head_q.cos(), head_q.sin()], dim=-1)  # [A, 2]

        x_t = scene_enc['x_a'].reshape(-1, D)  # [A*Th, D]
        x_pl = scene_enc['x_pl'][:, Th - 1]  # [Pl, D]
        q = self.query_emb.weight.repeat(A, 1)  # [A, D]
        m = self.mode_emb.weight.repeat(A, 1)  # [A*M, D]

        mask_src = data['agent']['valid_mask'][:, :Th].contiguous()  # [A, Th]
        mask_src[:, :Th - self.num_t2m_steps] = False  # [A, Th]
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True)  # [A, 1]

        # config t2q edges (temporal reasoning) {
        pos_t = data['agent']['position'][:, :Th, :Ci].reshape(-1, Ci)  # [A*Th, C]
        head_t = data['agent']['heading'][:, :Th].reshape(-1)  # [A*Th]
        edge_index_t2q = bipartite_dense_to_sparse(
            mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))  # [2, A*Th.1]
        rel_pos_t2m = pos_t[edge_index_t2q[0]] - pos_q[edge_index_t2q[1]]  # [A*Th.1, 2]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2q[0]] - head_q[edge_index_t2q[1]])
        r_t2q = torch.stack([
            torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_q[edge_index_t2q[1]],
                                     nbr_vector=rel_pos_t2m[:, :2]), rel_head_t2m,
            (edge_index_t2q[0] % Th) - Th + 1
        ],
                            dim=-1)  # [A*Th.1, 4]
        r_t2q = self.r_t2m_emb(continuous_inputs=r_t2q,
                               categorical_embs=None)  # [A*Th.1, D]
        # }

        # config pl2q edges (map guidance) {
        pos_pl = data['map_polygon']['position'][:, :Ci]  # [Pl, C]
        orient_pl = data['map_polygon']['orientation']  # [Pl]
        edge_index_pl2q = radius(
            x=pos_q[:, :2],  # [A, C]
            y=pos_pl[:, :2],  # [Pl, C]
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2q = edge_index_pl2q[:, mask_dst[edge_index_pl2q[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2q[0]] - pos_q[edge_index_pl2q[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2q[0]] -
                                     head_q[edge_index_pl2q[1]])
        r_pl2q = torch.stack([
            torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_q[edge_index_pl2q[1]],
                                     nbr_vector=rel_pos_pl2m[:, :2]), rel_orient_pl2m
        ],
                             dim=-1)
        r_pl2q = self.r_pl2m_emb(continuous_inputs=r_pl2q, categorical_embs=None)
        # }

        # config q2q edges (spatial reasoning) {
        edge_index_q2q = radius_graph(
            x=pos_q[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_q2q = edge_index_q2q[:, mask_src[:, -1][edge_index_q2q[0]] &
                                        mask_dst[edge_index_q2q[1], 0]]
        rel_pos_q2q = pos_q[edge_index_q2q[0]] - pos_q[edge_index_q2q[1]]
        rel_head_a2m = wrap_angle(head_q[edge_index_q2q[0]] - head_q[edge_index_q2q[1]])
        r_q2q = torch.stack([
            torch.norm(rel_pos_q2q[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(ctr_vector=head_vector_q[edge_index_q2q[1]],
                                     nbr_vector=rel_pos_q2q[:, :2]), rel_head_a2m
        ],
                            dim=-1)
        r_q2q = self.r_a2m_emb(continuous_inputs=r_q2q, categorical_embs=None)
        # }

        # config q2m edges (multi-mode readout) {
        edge_index_q2m = torch.stack([
            torch.arange(A, device=dev).repeat_interleave(M),
            torch.arange(A * M, device=dev)
        ])
        edge_index_q2m = edge_index_q2m[:, mask_dst[edge_index_q2m[1] // M, 0]]
        # }

        # config m2m edges (self-attention between modes) {
        mask_dst_m = mask_dst.repeat(1, M)  # [A, M]
        edge_index_m2m = dense_to_sparse(
            mask_dst_m.unsqueeze(2) & mask_dst_m.unsqueeze(1))[0]
        # }

        # propose {
        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * Trec
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * Trec
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * Trec
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * Trec
        for t in range(Trec):
            for i in range(self.num_layers):
                q = self.t2q_propose_attn_layers[i]((x_t, q), r_t2q, edge_index_t2q)
                q = self.pl2q_propose_attn_layers[i]((x_pl, q), r_pl2q, edge_index_pl2q)
                q = self.a2q_propose_attn_layers[i]((q, q), r_q2q, edge_index_q2q)
            q = self.q_out_norm[t](q)
    
            m = m.view(A * M, D)
            for i in range(self.num_layers):
                m = self.q2m_propose_attn_layers[i]((q, m), None, edge_index_q2m)
                m = self.m2m_propose_attn_layers[i](m, None, edge_index_m2m)
            
            m = self.m_propose_out_norm[t](m)
            m = m.view(A, M * D)
            q = self.m2q_mlp(m) + q.view(A, D)
            m = m.view(A, M, D)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)
            scales_propose_pos[t] = self.to_scale_propose_pos(m)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)
        loc_propose_pos = torch.cumsum(torch.cat(locs_propose_pos,
                                                 dim=-1).view(-1, M, Tf, Co),
                                       dim=-2)
        scale_propose_pos = torch.cumsum(F.elu_(
            torch.cat(scales_propose_pos, dim=-1).view(-1, M, Tf, Co), alpha=1.0) + 1.0,
                                         dim=-2) + 0.1
        # }
        
        if self.output_head:
            loc_propose_head = torch.cumsum(
                torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) *
                math.pi,
                dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(
                F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                dim=-2) + 0.02)
        else:
            loc_propose_head = loc_propose_pos.new_zeros(
                (loc_propose_pos.size(0), M, Tf, 1))
            conc_propose_head = scale_propose_pos.new_zeros(
                (scale_propose_pos.size(0), M, Tf, 1))
        
        # re-embed {
            
        if self.refine_enabled:
            if self.output_head:
                m = self.y_emb(
                    torch.cat(
                        [loc_propose_pos.detach(),
                        wrap_angle(loc_propose_head.detach())],
                        dim=-1).view(-1, Co + 1))
            else:
                m = self.y_emb(loc_propose_pos.detach().view(-1, Co))
            m = m.reshape(-1, Tf, self.hidden_dim).transpose(0, 1)
            m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        # }

        # refine {
        if self.refine_enabled:
            m = m.view(A * M, D)
            for i in range(self.num_layers):
                m = self.q2m_refine_attn_layers[i]((q, m), None, edge_index_q2m)
            m = self.m_refine_out_norm(m)
            m = m.view(A, M, D)

            loc_refine_pos = self.to_loc_refine_pos(m).view(-1, M, Tf, Co)
            loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
            scale_refine_pos = F.elu_(self.to_scale_refine_pos(m).view(-1, M, Tf, Co),
                                    alpha=1.0) + 1.0 + 0.1
            if self.output_head:
                loc_refine_head = torch.tanh(
                    self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
                loc_refine_head = loc_refine_head + loc_propose_head.detach()
                conc_refine_head = 1.0 / (
                    F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
            else:
                loc_refine_head = loc_refine_pos.new_zeros(
                    (loc_refine_pos.size(0), M, Tf, 1))
                conc_refine_head = scale_refine_pos.new_zeros(
                    (scale_refine_pos.size(0), M, Tf, 1))
        # }

        pi = self.to_pi(m.detach() if self.detach_m else m).squeeze(-1)

        if self.refine_enabled:
            return {
                'loc_propose_pos': loc_propose_pos,
                'scale_propose_pos': scale_propose_pos,
                'loc_propose_head': loc_propose_head,
                'conc_propose_head': conc_propose_head,
                'loc_refine_pos': loc_refine_pos,
                'scale_refine_pos': scale_refine_pos,
                'loc_refine_head': loc_refine_head,
                'conc_refine_head': conc_refine_head,
                'pi': pi,
            }
        else:
            return {
                'loc_propose_pos': loc_propose_pos,
                'scale_propose_pos': scale_propose_pos,
                'loc_propose_head': loc_propose_head,
                'conc_propose_head': conc_propose_head,
                # 'loc_refine_pos': loc_propose_pos.detach(),
                # 'scale_refine_pos': scale_propose_pos.detach(),
                # 'loc_refine_head': loc_propose_head.detach(),
                # 'conc_refine_head': conc_propose_head.detach(),
                'pi': pi,
            }
