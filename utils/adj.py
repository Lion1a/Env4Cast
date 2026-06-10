# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta
# import math
#
# def diffgraph():
#     # 加载数据
#     latitudes = np.load('/data/data_1/zxy/Env4Cast-main/latitudes.npy')     # (N,)
#     longitudes = np.load('/data/data_1/zxy/Env4Cast-main/longitudes.npy')   # (N,)
#
#     # 取前10000个空间点（按列）
#     latitudes = latitudes[:, :10000]  # shape = (T, 10000)
#     longitudes = longitudes[:, :10000]  # shape = (T, 10000)
#
#     # 构建邻接矩阵，直接存储距离值
#     adj_haversine = build_adjacency_matrix(longitudes, latitudes)
#
#     return adj_haversine
#
#
# # Haversine 距离计算函数
# def haversine(lat1, lon1, lat2, lon2):
#     R = 6371  # 地球半径，单位 km
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     delta_phi = math.radians(lat2 - lat1)
#     delta_lambda = math.radians(lon2 - lon1)
#
#     a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#
#     return R * c  # 返回距离，单位 km
#
#
# def build_adjacency_matrix(longitudes, latitudes):
#     n = len(longitudes)
#     adj_matrix = np.zeros((n, n))  # 初始化邻接矩阵
#
#     for i in range(n):
#         for j in range(i + 1, n):
#             lat1, lon1 = latitudes[i], longitudes[i]
#             lat2, lon2 = latitudes[j], longitudes[j]
#
#             distance = haversine(lat1, lon1, lat2, lon2)
#
#             # 将计算出的距离存入矩阵
#             adj_matrix[i, j] = adj_matrix[j, i] = distance  # 因为是无向图，填充对称位置
#
#     return adj_matrix
#
#
#
