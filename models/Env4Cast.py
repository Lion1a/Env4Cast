import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.AMS import AMS
from layers.Wave import WaveletLayer
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
from layers.gcn import CrossGraphLearningModel
import torch.nn.functional as F

from functools import reduce
from operator import mul

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.beta=0.01
        self.c_w=0.001
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.wavelet_layer = WaveletLayer(pred_len=0, k=1)
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.attention = nn.MultiheadAttention(embed_dim=3000, num_heads=3, batch_first=False)
        self.linear_transform = nn.Linear(2000, 3000)
        self.linear_layer = nn.Linear(3000, 2000)
        self.projections2 = nn.Linear(12, 12)


        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.batch_norm = configs.batch_norm
        self.GLM = CrossGraphLearningModel(in_features=12, out_features=256)

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )


    def min_max_normalization(self,matrix):
        matrix = torch.tensor(matrix).cuda(5)
        mean = torch.mean(matrix)
        std = torch.std(matrix)
        return (matrix - mean) / std

    def compute_angle_matrix(self,pos, velo, eps=1e-8):
        N = pos.shape[0]
        angle_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # 位移向量：从 i 指向 j
                d_ij = pos[j] - pos[i]  # shape (2,)
                v_i = velo[i]  # i 点的速度向量

                # 点积 & 模长乘积
                dot = np.dot(v_i, d_ij)
                norm_product = np.linalg.norm(v_i) * np.linalg.norm(d_ij) + eps

                cos_xi = dot / norm_product
                angle_matrix[i, j] = cos_xi
        return angle_matrix

    def get_reciprocal_laplacian(self,laplacian_diff):
        """
        处理三维拉普拉斯矩阵的倒数
        """
        epsilon = 1e-12
        # 添加微小常数避免除零
        protected_matrix = laplacian_diff + epsilon
        reciprocal = np.where(np.abs(laplacian_diff) > epsilon,
                              1.0 / protected_matrix,
                              0)
        return reciprocal
    def normalize_to_range(self,tensor, min_val=0.0, max_val=0.1):
        t_min = torch.min(tensor)
        t_max = torch.max(tensor)
        # 避免除以 0
        if t_max - t_min < 1e-8:
            return torch.zeros_like(tensor) + min_val
        return (tensor - t_min) / (t_max - t_min) * (max_val - min_val) + min_val

    def forward(self, x,velo,adj_haversine, laplacian_diff, laplacian_heat):
        # print(x.shape, velo.shape)
        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))

        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss
        low_T,high_T = self.wavelet_layer(x)
        # print(low_T.shape, x.shape)
        low_V,high_V = self.wavelet_layer(velo)

        velo = velo.cpu()
        velo = velo[0][0]
        velo_matrix = velo[:, np.newaxis]
        # angle = self.compute_angle_matrix(velo, velo_matrix)
        G_ns = np.maximum(0, velo_matrix/adj_haversine)
        G_ns[torch.isinf(G_ns)] = 0
        # degree_matrix = np.diag(np.sum(adj_haversine, axis=1))
        degree_ns = torch.diag(torch.sum(G_ns, axis=1))
        G_ns = torch.tensor(G_ns).cuda(5)

        degree_ns = torch.tensor(degree_ns).cuda(5)

        # 计算拉普拉斯矩阵 L = D - A
        laplacian_ns = degree_ns - G_ns

        # laplacian_diff = self.get_reciprocal_laplacian(laplacian_diff)
        laplacian_diff = self.min_max_normalization(laplacian_diff)
        laplacian_heat = self.min_max_normalization(laplacian_heat)
        laplacian_ns = self.min_max_normalization(laplacian_ns)
        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        low_T = low_T.reshape(batch_size, 12, 2000)
        high_V = high_V.reshape(batch_size, 12, 2000)

        laplacian_ns = torch.nan_to_num(laplacian_ns, nan=0)

        result1_norm = torch.matmul(low_T.to(device).double(), laplacian_diff.to(device).double())
        result2_norm = torch.matmul(high_V.to(device).double(), laplacian_ns.to(device).double())
        result3_norm = torch.matmul(low_T.to(device).double(), laplacian_heat.to(device).double())

        result1_norm = self.normalize_to_range(result1_norm, 0.0, 1)
        result2_norm = self.normalize_to_range(result2_norm, 0.0, 1)
        result3_norm = self.normalize_to_range(result3_norm, 0.0, 1)
        # result2_norm = result2_norm.unsqueeze(0).unsqueeze(0)
        # result2_norm = result2_norm.expand(batch_size, 12, 5000)


        result1_norm = self.linear_transform(result1_norm.float())  # Shape: (batch_size, seq_len, mapped_dim)
        result2_norm = self.linear_transform(result2_norm.float())  # Shape: (batch_size, seq_len, mapped_dim)
        result3_norm = self.linear_transform(result3_norm.float())  # Shape: (batch_size, seq_len, mapped_dim)


        result1_norm = result1_norm.permute(1, 0, 2)
        result2_norm = result2_norm.permute(1, 0, 2)
        result3_norm = result3_norm.permute(1, 0, 2)

        output1, _ = self.attention(result1_norm, result2_norm, result3_norm)
        output2, _ = self.attention(result2_norm, result3_norm, result1_norm)
        output3, _ = self.attention(result3_norm, result1_norm, result2_norm)

        result1_norm = self.linear_layer(output1)  # Shape: (batch_size, seq_len, mapped_dim)
        result2_norm = self.linear_layer(output2)  # Shape: (batch_size, seq_len, mapped_dim)
        result3_norm = self.linear_layer(output3)  # Shape: (batch_size, seq_len, mapped_dim)

        result1_norm = result1_norm.permute(1, 0, 2)
        result2_norm = result2_norm.permute(1, 0, 2)
        result3_norm = result3_norm.permute(1, 0, 2)
        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        out2 =self.beta*(result1_norm + result2_norm + result3_norm)
        out2_flattened = out2.reshape(-1, 12)
        out2_mapped = self.projections2(out2_flattened)
        out2 = out2_mapped.reshape(4, 2000, 12).permute(0, 2, 1)  # → (4, 12, 2000)
        out_3 = out + out2

        out = out.permute(0, 2, 1)
        out2 = out2.permute(0, 2, 1)
        B, N, T = out.shape
        out_flat = out.reshape(B * N, T)  # (B*N, T)
        out2_flat = out2.reshape(B * N, T)  # (B*N, T)
        out_flat = F.normalize(out_flat, dim=1)
        out2_flat = F.normalize(out2_flat, dim=1)
        similarity_matrix = torch.matmul(out_flat, out2_flat.T)
        labels = torch.arange(B * N).to(out.device)
        contrastive_loss = F.cross_entropy(similarity_matrix / 0.1, labels)

        return out_3, balance_loss, self.c_w*contrastive_loss


