import os, sys, torch, heapq
import numpy as np
import torch.nn.functional as F

sys.path.append("/Users/lyw/projects/ECNU/QoS_Prediction")
from models.base import ModelBase
from torch import nn
from torch_geometric.nn import GCNConv
from utils.data_process import *


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  # 支持向量
        output = torch.matmul(adj, support) + self.bias  # 邻接矩阵的转置乘以支持向量，添加偏置项
        return output


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(in_channels, hidden_channels)
        self.conv2 = GraphConvolution(hidden_channels, out_channels)
        # self.norm_layer = nn.LayerNorm(in_channels)
        self.dropout = 0.2

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, adj))
        # x = self.norm_layer(x)
        return x


class Glbp(nn.Module):
    def __init__(self, rt_data, adj_matrix, user_data, item_data, latent_dim, layers) -> None:
        super(Glbp, self).__init__()
        self.subgraph_size, self.num_user, self.num_item = 20, rt_data.row_n, rt_data.col_n
        self.rt_data, self.adj_matrix, self.user_data, self.item_data = rt_data, adj_matrix, user_data, item_data
        self.user_infos = [self.user_data.query(id) for id in range(self.num_user)]
        self.item_infos = [self.item_data.query(id) for id in range(self.num_item)]

        self.uid_embedding = nn.Embedding(num_embeddings=rt_data.row_n, embedding_dim=latent_dim)
        self.sid_embedding = nn.Embedding(num_embeddings=rt_data.col_n, embedding_dim=latent_dim)

        self.init_ufeature = nn.Linear(len(user_data.enabled_columns), latent_dim)  # user特征embedding层
        self.init_ifeature = nn.Linear(len(item_data.enabled_columns), latent_dim)  # item特征embedding层

        # GCN模块
        self.gcn = GCN(latent_dim, latent_dim * 2, latent_dim)

        # 输出模块的MLP网络
        self.output_mlp = nn.Sequential(*[nn.Sequential(nn.Linear(layers[i], layers[i + 1]), nn.ReLU()) for i in range(len(layers) - 1)])

    def forward(self, user_indexes, item_indexes):
        # ========================================特征初始化========================================
        id_vec = torch.cat([self.uid_embedding(user_indexes), self.sid_embedding(item_indexes)], dim=-1)
        user_features = F.normalize(self.init_ufeature(torch.stack(self.user_infos)), p=2, dim=1)
        item_features = F.normalize(self.init_ifeature(torch.stack(self.item_infos)), p=2, dim=1)
        adj_x = torch.cat([user_features, item_features], dim=0)
        # ft_vec = torch.cat([user_features[user_indexes, :], item_features[item_indexes, :]], dim=-1)

        # ======================================== GCN部分 ========================================
        subgraph_adjs, subgraph_xs, subgraph_roots = self.build_GCN_input(user_indexes, item_indexes, adj_x)
        gcn_xs = self.gcn(subgraph_xs, subgraph_adjs)
        ft_vec = torch.stack([gcn_xs[i][subgraph_roots[i], :] for i in range(len(gcn_xs))])
        ft_vec = torch.cat([ft_vec[:, 0, :], ft_vec[:, 1, :]], dim=1)

        # ========================================模型输出层========================================
        x = 0.6 * id_vec + 0.4 * ft_vec
        x = self.output_mlp(x)
        return x

    # 构造GCN的输入 子图邻接矩阵 与 特征向量
    def build_GCN_input(self, user_indexes, item_indexes, adj_x):
        subgraph_adjs, subgraph_xs, subgraph_roots = [], [], []
        for i in range(len(user_indexes)):
            row_id, col_id = int(user_indexes[i]), int(item_indexes[i]) + self.rt_data.row_n  # 用户和服务对应的行列
            sample_ids, subgraph_adj = self.subgraph_sampling(row_id, col_id)  # 以[rowid, colid]为root对邻接矩阵adj_mat进行子图采样
            subgraph_xs.append(adj_x[sample_ids, :])
            subgraph_adjs.append(torch.tensor(subgraph_adj, dtype=torch.float32))
            subgraph_roots.append(torch.tensor([sample_ids.index(row_id), sample_ids.index(col_id)], dtype=torch.int32))
        return torch.stack(subgraph_adjs), torch.stack(subgraph_xs), torch.stack(subgraph_roots)

    # 以[rowid, colid]为root对邻接矩阵adj_mat进行子图采样
    def subgraph_sampling(self, row_id, col_id):
        num_user, num_item = self.rt_data.row_n, self.rt_data.col_n
        sample_uids = random.sample(range(num_user), int(self.subgraph_size / 2))  # 随机采样用户
        sample_sids = random.sample(range(num_user, num_user + num_item), int(self.subgraph_size / 2))  # 随机采样服务
        if row_id not in sample_uids:
            sample_uids[0] = row_id
        if col_id not in sample_sids:
            sample_sids[0] = col_id
        sample_ids = sorted(sample_uids + sample_sids)
        subgraph_adj = self.adj_matrix[sample_ids, :][:, sample_ids]
        return sample_ids, subgraph_adj


class GlbpModel(ModelBase):
    def __init__(self, loss_fn, rt_data, adj_matrix, user_data, item_data, latent_dim, use_gpu=True) -> None:
        super().__init__(loss_fn, use_gpu)
        self.name = __class__.__name__
        self.model = Glbp(rt_data, adj_matrix, user_data, item_data, latent_dim, layers=[latent_dim * 2, 16, 8, 1])
        self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)
