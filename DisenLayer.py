from helper import *
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn
import torch.nn.functional as F

# 修改测试
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean, scatter_max
from torch_geometric.utils import softmax


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_uniform_(param)
    return param


class DisenLayer(MessagePassing):
    def __init__(self, edge_index, edge_type, in_channels, out_channels, num_rels, act=lambda x: x, params=None,
                 head_num=1):
        super(self.__class__, self).__init__(aggr='add', flow='target_to_source', node_dim=0)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.device = None
        self.head_num = head_num
        self.num_rels = num_rels

        # params for init
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm1d(self.p.num_factors * out_channels)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

        num_edges = self.edge_index.size(1) // 2
        if self.device is None:
            self.device = self.edge_index.device
        self.in_index, self.out_index = self.edge_index[:, :num_edges], self.edge_index[:, num_edges:]
        self.in_type, self.out_type = self.edge_type[:num_edges], self.edge_type[num_edges:]
        self.loop_index = torch.stack([torch.arange(self.p.num_ent), torch.arange(self.p.num_ent)]).to(self.device)
        self.loop_type = torch.full((self.p.num_ent,), 2 * self.num_rels, dtype=torch.long).to(self.device)
        num_ent = self.p.num_ent

        self.leakyrelu = nn.LeakyReLU(0.2)
        if self.p.att_mode == 'cat_emb' or self.p.att_mode == 'cat_weight':
            self.att_weight = get_param((1, self.p.num_factors, 2 * out_channels))
        else:
            self.att_weight = get_param((1, self.p.num_factors, out_channels))
        self.rel_weight = get_param((2 * self.num_rels + 1, self.p.num_factors, out_channels))
        self.loop_rel = get_param((1, out_channels))
        self.w_rel = get_param((out_channels, out_channels))

        self.global_rel_emb = nn.Parameter(torch.randn(num_rels, out_channels))
        self.w_rel_global = nn.Parameter(torch.randn(out_channels, out_channels))

    def forward(self, x, rel_embed, k_weights, mode):
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        global_rel_embed = torch.matmul(rel_embed, self.w_rel_global)

        edge_index = torch.cat([self.edge_index, self.loop_index], dim=1)
        edge_type = torch.cat([self.edge_type, self.loop_type])
        out = self.propagate(edge_index, size=None, x=x, edge_type=edge_type, rel_embed=rel_embed,
                             rel_weight=self.rel_weight, k_weights=k_weights)
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out.view(-1, self.p.num_factors * self.p.gcn_dim)).view(-1, self.p.num_factors, self.p.gcn_dim)
        entity1 = out if self.p.no_act else self.act(out)
        return entity1, global_rel_embed


    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_type, rel_embed, rel_weight, k_weights_i, k_weights_j):

        rel_embed = torch.index_select(rel_embed, 0, edge_type)
        rel_weight = torch.index_select(rel_weight, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_embed, rel_weight)
        max_index = self.global_rel_emb.size(0) - 1
        clamped_edge_type = torch.clamp(edge_type, 0, max_index)
        global_rel_embed = torch.index_select(self.global_rel_emb, 0, clamped_edge_type)
        xj_rel = xj_rel + global_rel_embed.unsqueeze(1)
        alpha = self._get_attention(edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, xj_rel)
        alpha = self.drop(alpha)
        return k_weights_j * k_weights_i * alpha * xj_rel

    def update(self, aggr_out):
        return aggr_out

    def _get_attention(self, edge_index_i, edge_index_j, x_i, x_j, rel_embed, rel_weight, mes_xj):
        sub_rel_emb = x_i * rel_weight
        obj_rel_emb = x_j * rel_weight
        alpha = self.leakyrelu(torch.einsum('ekf,ekf->ek', [sub_rel_emb, obj_rel_emb]))
        num_nodes = self.p.num_ent
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)

        return alpha.unsqueeze(2)

    def rel_transform(self, ent_embed, rel_embed, rel_weight, opn=None):
        if opn is None:
            opn = self.p.opn
        if opn == 'sub':
            trans_embed = ent_embed * rel_weight - rel_embed.unsqueeze(1)
        elif opn == 'mult':
            trans_embed = (ent_embed * rel_embed.unsqueeze(1)) * rel_weight
        elif opn == 'cross':
            trans_embed = ent_embed * rel_embed.unsqueeze(1) * rel_weight + ent_embed * rel_weight
        else:
            raise NotImplementedError

        return trans_embed

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
