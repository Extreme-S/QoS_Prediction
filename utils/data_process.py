import sys, torch, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine, cdist
from utils.util import *
from const import *


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 初始化节点特征
def read_ui_data(dataPath, dim):
    # node_vectors = []
    # user
    with open(dataPath + "userlist.txt", "r") as user_file:
        lines = user_file.readlines()
    lines = lines[2:]  # 跳过前两行（标题和分隔线）
    user_data = [dict(zip(user_headers, line.strip().split("\t"))) for line in lines if line.strip()]
    # node_vectors.extend(build_user_vectors(user_data, dim))

    # service
    with open(dataPath + "wslist.txt", "r") as item_file:
        lines = item_file.readlines()
    lines = lines[2:]  # 跳过前两行（标题和分隔线）
    item_data = [dict(zip(service_headers, line.strip().split("\t"))) for line in lines if line.strip()]
    # node_vectors.extend(build_item_vectors(item_data, dim))

    return user_data, item_data


# 提取用户特征向量
def build_user_vectors(data, dim):
    user_ids = [int(item["User ID"]) for item in data]
    ip_addresses = [item["IP Address"] for item in data]
    countries = [item["Country"] for item in data]
    as_info = [item["AS"] for item in data]
    latitudes = [float(item["Latitude"]) for item in data]
    longitudes = [float(item["Longitude"]) for item in data]

    # one-hot编码离散数据
    encoded_countries = LabelEncoder().fit_transform(countries)
    encoded_as_info = LabelEncoder().fit_transform(as_info)

    # 构建用户特征向量
    feature_vectors = [
        torch.tensor([user_id, country, as_num, lat, lon], dtype=torch.float32)
        for user_id, country, as_num, lat, lon in zip(
            user_ids,
            encoded_countries,
            encoded_as_info,
            latitudes,
            longitudes,
        )
    ]
    fcn = nn.Linear(5, dim)  # 定义全连接层
    return fcn(torch.stack(feature_vectors))


# 提取服务特征向量
def build_item_vectors(data, dim):
    item_ids = [int(item["Service ID"]) for item in data]
    wsdl_addresses = [item["WSDL Address"] for item in data]
    service_providers = [item["Service Provider"] for item in data]
    ip_addresses = [item["IP Address"] for item in data]
    countries = [item["Country"] for item in data]
    ip_numbers = [item["IP No."] for item in data]
    as_info = [item["AS"] for item in data]
    latitudes = [float(item["Latitude"]) if item["Latitude"] != "null" else 0.0 for item in data]
    longitudes = [float(item["Longitude"]) if item["Longitude"] != "null" else 0.0 for item in data]

    # one-hot编码离散数据
    encoded_countries = LabelEncoder().fit_transform(countries)
    encoded_as_info = LabelEncoder().fit_transform(as_info)

    # 构建用户特征向量
    feature_vectors = [
        # torch.tensor([item_id, "ip", country, "ip_numbers", as_num, lat, lon], dtype=torch.float32)
        torch.tensor([item_id, country, as_num], dtype=torch.float32)
        for item_id, wsdl, sp, ip, country, ip_numbers, as_num, lat, lon in zip(
            item_ids,
            wsdl_addresses,
            service_providers,
            ip_addresses,
            encoded_countries,
            ip_numbers,
            encoded_as_info,
            latitudes,
            longitudes,
        )
    ]
    fcn = nn.Linear(3, dim)  # 定义全连接层
    return fcn(torch.stack(feature_vectors))


# 读取rt矩阵
def read_rtMatrix(data_path):
    with open(data_path, "r") as user_file:
        lines = user_file.readlines()
    rtMatrix = [line.strip().split("\t") for line in lines if line.strip()]
    return np.array(rtMatrix, dtype=float)


# 读取用户和服务的辅助信息
def read_us_info(dataPath):
    with open(dataPath + "userlist.txt", "r") as user_file:
        lines = user_file.readlines()
    lines = lines[2:]  # 跳过前两行（标题和分隔线）
    user_data = [dict(zip(user_headers, line.strip().split("\t"))) for line in lines if line.strip()]

    with open(dataPath + "wslist.txt", "r") as item_file:
        lines = item_file.readlines()
    lines = lines[2:]  # 跳过前两行（标题和分隔线）
    item_data = [dict(zip(service_headers, line.strip().split("\t"))) for line in lines if line.strip()]

    return user_data, item_data


# 构建位置权重的邻接矩阵
def build_adj_matrix2(data_path):
    rtMatrix = read_rtMatrix(data_path + "rtMatrix.txt")
    users, items = read_us_info(data_path)
    # 计算距离
    for i in range(len(rtMatrix)):
        for j in range(len(rtMatrix[0])):
            lat1, lon1, lat2, lon2 = users[i]["Latitude"], users[i]["Longitude"], items[j]["Latitude"], items[j]["Longitude"]
            if lat1 == "null" or lon1 == "null" or lat2 == "null" or lon2 == "null":  # 位置信息缺失
                rtMatrix[i][j] = -2  # 经纬度缺失
                continue
            rtMatrix[i][j] = haversine(float(lat1), float(lon1), float(lat2), float(lon2))
    valid_value = rtMatrix[rtMatrix >= 0]
    max_value, min_value, mean_value = np.max(valid_value), np.min(valid_value), np.mean(valid_value)  # max，min，mean
    rtMatrix = np.where(rtMatrix >= 0, (rtMatrix - min_value) / (max_value - min_value), rtMatrix)  # 等比缩放
    rtMatrix = np.where(rtMatrix == -2, mean_value, rtMatrix)  # 替换缺失值为平均值

    # 完善邻接矩阵，0索引处为填充值
    num_user, num_item = len(rtMatrix), len(rtMatrix[0])
    node_matrix = np.hstack((np.full((num_user, num_user), -1), rtMatrix))
    node_matrix = np.vstack((node_matrix, np.full((num_item, num_user + num_item), -1)))
    node_matrix = np.hstack((np.full((num_user + num_item, 1), -1), node_matrix))
    node_matrix = np.vstack((np.full((1, num_user + num_item + 1), -1), node_matrix))

    # 按照斜对角线对称
    nozero_indexs = np.argwhere(node_matrix != -1)
    for i, j in nozero_indexs:
        node_matrix[j][i] = node_matrix[i][j]

    return node_matrix


# 根据节点从邻接矩阵提取子图 子图用户数
def extract_subgraph2(row_id, col_id, adj, rt_data, sample_u_num, sample_i_num):
    # 抽取采样节点
    # valid_col_indexs, valid_row_indexs = np.where(adj[row_id, :] >= 0)[0], np.where(adj[:, col_id] >= 0)[0]
    # sample_row_indexs = random.sample(valid_row_indexs.tolist(), min(subgraph_i_num, len(valid_row_indexs)))
    # sample_col_indexs = random.sample(valid_col_indexs.tolist(), min(subgraph_u_num, len(valid_col_indexs)))
    # if row_id not in sample_row_indexs:
    #     if len(sample_row_indexs) < subgraph_u_num:
    #         sample_row_indexs.append(row_id)
    #     else:
    #         sample_row_indexs[0] = row_id
    # if col_id not in sample_col_indexs:
    #     if len(sample_col_indexs) < subgraph_i_num:
    #         sample_col_indexs.append(col_id)
    #     else:
    #         sample_col_indexs[0] = col_id
    # sample_indexs = sorted(sample_row_indexs + sample_col_indexs)
    u_range, i_range = range(1, rt_data.row_n + 1), range(1 + rt_data.row_n, rt_data.col_n + rt_data.row_n + 1)
    sample_indexs = random.sample(u_range, sample_u_num) + random.sample(i_range, sample_i_num)
    if row_id not in sample_indexs:
        sample_indexs[0] = row_id
    if col_id not in sample_indexs:
        sample_indexs[len(sample_indexs) - 1] = col_id
    sample_indexs = sorted(sample_indexs)
    subgraph = adj[sample_indexs, :][:, sample_indexs]
    return sample_indexs, torch.from_numpy(subgraph).type(torch.float32)


# 构建邻接矩阵（相似度感知）
def build_adj_matrix(data_path):
    rat_mat = read_rtMatrix(data_path + "rtMatrix.txt")  # qos 矩阵
    u_sim_mat, i_sim_mat = build_sim_mat(rat_mat)  # 获取相似度矩阵

    # 构建邻接矩阵
    num_user, num_item = len(rat_mat), len(rat_mat[0])
    adj_mat = np.zeros((num_user + num_item, num_user + num_item))
    adj_mat[:num_user, num_user:] = (rat_mat > 0).astype(int)  # 调用关系
    adj_mat += adj_mat.T - np.diag(np.diag(adj_mat))  # 按照对角线对称
    adj_mat[:num_user, :num_user], adj_mat[num_user:, num_user:] = u_sim_mat, i_sim_mat  # 加入同类相似性关系
    np.fill_diagonal(adj_mat, float(1))
    return adj_mat


# 获取用户/服务的 相似度矩阵
def build_sim_mat(rat_mat):
    u_num, i_num = len(rat_mat), len(rat_mat[0])
    user_sim_mat = 1 - cdist(rat_mat, rat_mat, metric="cosine")
    item_sim_mat = 1 - cdist(rat_mat.T, rat_mat.T, metric="cosine")

    # 确保对角线元素为0（因为自身与自身的相似度为0）
    np.fill_diagonal(user_sim_mat, 1)
    np.fill_diagonal(item_sim_mat, 1)
    return user_sim_mat, item_sim_mat
