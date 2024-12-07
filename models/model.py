import os, sys, torch, heapq
import numpy as np
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import GCNConv

sys.path.append("/Users/lyw/projects/ECNU/QoS_Prediction")
from models.base import ModelBase
from torch import nn
from torch_geometric.nn import GCNConv
from utils.data_process import *


class Attn(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(Attn, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)

    def forward(self, query, value, key):
        # 假设value, key, query的形状都是 (n, F)，其中n是序列长度（或批次大小），F是特征维度
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output


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
    def __init__(self, in_channels, hidden_channels, out_channels, subgraph_size):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(subgraph_size)
        self.conv2 = GraphConvolution(hidden_channels, out_channels)
        self.norm_layer = nn.LayerNorm(out_channels)
        self.dropout = 0.2

    def forward(self, x, adj):
        x = self.bn1(self.conv1(x, adj))
        x = F.leaky_relu(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.conv2(x, adj)
        return x


class Glbp(nn.Module):
    def __init__(self, rt_data, adj_matrix, user_data, item_data, latent_dim, subgraph_size, alpha) -> None:
        super(Glbp, self).__init__()
        self.rt_data, self.adj_matrix, self.user_data, self.item_data = rt_data, adj_matrix, user_data, item_data
        self.num_user, self.num_item, self.subgraph_size, self.alpha = rt_data.row_n, rt_data.col_n, subgraph_size, alpha
        self.user_infos = [self.user_data.query(id) for id in range(self.num_user)]
        self.item_infos = [self.item_data.query(id) for id in range(self.num_item)]

        self.uid_embedding = nn.Embedding(num_embeddings=rt_data.row_n, embedding_dim=latent_dim)
        self.sid_embedding = nn.Embedding(num_embeddings=rt_data.col_n, embedding_dim=latent_dim)

        self.init_ufeature = nn.Linear(len(user_data.enabled_columns), latent_dim)  # user特征embedding层
        self.init_ifeature = nn.Linear(len(item_data.enabled_columns), latent_dim)  # item特征embedding层

        # 注意力模块
        self.user_attn, self.item_attn = Attn(latent_dim, 2), Attn(latent_dim, 2)

        # GCN模块
        self.gcn = GCN(latent_dim, latent_dim, latent_dim, self.subgraph_size)

        # 输出模块的MLP网络
        layers = [latent_dim * 2, 16, 8, 1]
        self.output_mlp = nn.Sequential(*[nn.Sequential(nn.Linear(layers[i], layers[i + 1]), nn.ReLU()) for i in range(len(layers) - 1)])

    def forward(self, user_indexes, item_indexes):
        # ========================================特征初始化========================================
        user_ids, item_ids = self.uid_embedding(torch.arange(self.num_user)), self.sid_embedding(torch.arange(self.num_item))
        id_vec = torch.cat([user_ids[user_indexes, :], item_ids[item_indexes, :]], dim=-1)
        user_features = F.normalize(self.init_ufeature(torch.stack(self.user_infos)), p=2, dim=1)
        item_features = F.normalize(self.init_ifeature(torch.stack(self.item_infos)), p=2, dim=1)
        # x = self.a_id * torch.cat([user_ids, item_ids], dim=0) + (1 - self.a_id) * torch.cat([user_features, item_features], dim=0)
        x = torch.cat([user_features, item_features], dim=0)
        subgraph_adjs, subgraph_xs, subgraph_roots = self.build_GCN_input(user_indexes, item_indexes, x)

        # ========================================注意力部分========================================
        # user_vec, item_vec = self.user_attn(user_ids, user_ids, user_ids), self.item_attn(item_ids, item_ids, item_ids)
        # att_res_vec = torch.cat([user_vec[user_indexes], item_vec[item_indexes]], dim=-1)

        # ======================================== GCN部分 ========================================
        gcn_xs = self.gcn(subgraph_xs, subgraph_adjs)
        gcn_vec = torch.stack([gcn_xs[i][subgraph_roots[i], :] for i in range(len(gcn_xs))])
        gcn_res_vec = torch.cat([gcn_vec[:, 0, :], gcn_vec[:, 1, :]], dim=1)

        # ========================================模型输出层========================================
        res_vec = self.alpha * id_vec + (1 - self.alpha) * gcn_res_vec
        return self.output_mlp(res_vec)

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
        sample_ids = sorted(sample_uids + sample_sids)  # 采样的ids
        subgraph_adj = self.adj_matrix[sample_ids, :][:, sample_ids]  # 提取子图
        subgraph_adj[subgraph_adj == 1] = 0.2  # 配置调用关系的权重
        np.fill_diagonal(subgraph_adj, 1)  # 自身权重赋值1
        return sample_ids, subgraph_adj


class GlbpModel(ModelBase):
    def __init__(self, loss_fn, rt_data, adj_matrix, user_data, item_data, latent_dim, subgraph_size, alpha, use_gpu=True) -> None:
        super().__init__(loss_fn, use_gpu)
        self.name = __class__.__name__
        self.model = Glbp(rt_data, adj_matrix, user_data, item_data, latent_dim, subgraph_size, alpha)
        self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)
