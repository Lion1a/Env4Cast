# import torch
# import torch.nn.functional as F
# import torch.nn as nn
#
#
# class CrossGraphLearningModel(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(CrossGraphLearningModel, self).__init__()
#         self.gcn = GraphConvolutionLayer(in_features, out_features)  # GCN 图卷积层
#
#     def forward(self, G_heat, G_diff, G_adv, features):
#         # 初始化权重向量 (例如 alpha, beta, gamma)
#         theta = torch.randn(3, device=G_heat.device)  # 初始化权重为随机值
#
#         # 应用 Softmax 来归一化权重
#         weights = F.softmax(theta, dim=0)  # 归一化权重，dim=0 表示对 3 个图的权重归一化
#
#         # 通过加权求和计算合并后的邻接矩阵
#         weighted_sum_adj = weights[0] * G_heat + weights[1] * G_diff + weights[2] * G_adv
#
#         # 使用加权后的邻接矩阵和节点特征进行图卷积
#         output = self.gcn(weighted_sum_adj, features)
#
#         return output
#
# class GraphConvolutionLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(GraphConvolutionLayer, self).__init__()
#         self.weight = nn.Parameter(torch.randn(in_features, out_features))  # 权重矩阵
#
#     def forward(self, adj, features):
#         # 图卷积操作：A @ X @ W
#         return torch.matmul(adj, torch.matmul(features, self.weight))

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))  # learnable W

    def forward(self, adj, features):
        # adj: (B, N, N)
        # features: (B, N, F_in)
        support = torch.matmul(features, self.weight)      # (B, N, F_out)
        out = torch.matmul(adj, support)                   # (B, N, F_out)
        return out

class CrossGraphLearningModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(CrossGraphLearningModel, self).__init__()
        self.gcn = GraphConvolutionLayer(in_features, out_features)

        # 初始化三个可学习的图融合权重（theta），使用 softmax 得到 [α, β, γ]
        self.theta = nn.Parameter(torch.randn(3))

    def forward(self, G_heat, G_diff, G_adv, features):
        """
        G_heat, G_diff, G_adv: (N, N)
        features: (B, T, N) → 每个样本是一个 (T=时间, N=节点) 的矩阵
        """
        B, T, N = features.shape

        # 转为 (B, N, T)，将每个节点的时间序列作为特征
        features = features.permute(0, 2, 1)  # → (B, N, T)

        # Expand 邻接矩阵成 batched 版本：→ (B, N, N)
        G_heat = G_heat.unsqueeze(0).expand(B, -1, -1)
        G_diff = G_diff.unsqueeze(0).expand(B, -1, -1)
        G_adv  = G_adv.unsqueeze(0).expand(B, -1, -1)

        # 权重归一化 (softmax)
        weights = F.softmax(self.theta, dim=0)  # [α, β, γ] ∈ [0,1], sum=1

        # 加权融合三个图
        adj_combined = (
            weights[0] * G_heat +
            weights[1] * G_diff +
            weights[2] * G_adv
        )  # shape: (B, N, N)

        # 图卷积
        out = self.gcn(adj_combined, features)  # → (B, N, out_features)
        return out

