import torch
import torch.nn as nn
import pywt
import numpy as np
from einops import rearrange, repeat


class WaveletLayer(nn.Module):
    def __init__(self, pred_len, k=None, low_freq=2, output_attention=False, wavelet='db4', level=2):
        super().__init__()
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_wavelet = self.wavelet_transform(x)

        low, high = self.select_high_low_freq(x_wavelet)

        low, high = self.inverse_wavelet_transform(low,high)

        return low, high

    def wavelet_transform(self, x):
        b, t, d = x.shape  # 获取输入数据的形状

        x_reshaped = x.view(b, -1)  # (b, t * d)

        x_wavelet = []

        for n in range(b):
            signal = x_reshaped[n].detach().cpu().numpy()

            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

            max_len = max([len(c) for c in coeffs])  # 计算最大长度

            coeffs_padded = []
            for c in coeffs:
                if c.ndim == 1:  # 如果是 1D 数组
                    coeffs_padded.append(np.pad(c, (0, max_len - len(c)), mode='constant'))
                elif c.ndim == 2:  # 如果是 2D 数组
                    coeffs_padded.append(np.pad(c, ((0, 0), (0, max_len - c.shape[1])), mode='constant'))
            # coeffs_padded: List[level+1] -> 每个是 (b, max_len)
            # 叠加成 (level+1, b, max_len)
            coeffs_stacked = np.stack(coeffs_padded, axis=0)
            x_wavelet.append(coeffs_stacked)  # (level+1, b, max_len)

        # x_wavelet: list of (level+1, b, max_len) * d → shape: (d, level+1, b, max_len)
        x_wavelet = np.stack(x_wavelet, axis=0)
        # print(x_wavelet.shape)
        # 转为张量 (d, level+1, b, max_len) → (b, level+1, d, max_len)
        # x_wavelet = torch.tensor(x_wavelet).float().permute(2, 1, 0, 3).to(x.device)

        return x_wavelet

    def select_high_low_freq(self, x_wavelet):

        b, l, d= x_wavelet.shape  # l = level+1

        low_freq_coeffs = x_wavelet[:, :self.low_freq,  :]  # 低频：最前面的近似层
        high_freq_coeffs = x_wavelet[:, -self.low_freq:,  :]  # 高频：最后的细节层

        return low_freq_coeffs, high_freq_coeffs

    def inverse_wavelet_transform(self, low_freq_coeffs, high_freq_coeffs):
        low = []
        high = []

        for b in range(low_freq_coeffs.shape[0]):
            low_coeff = low_freq_coeffs[b, :, :]
            high_coeff = high_freq_coeffs[b, :, :]
            # print("low_coeff",low_coeff.shape)

            coeffs = [low_coeff] + list(high_coeff)
            low_reconstructed = pywt.waverec([low_coeff] + [None] * (len(low_coeff) - 1), self.wavelet)
            high_reconstructed = pywt.waverec([None] * (len(high_coeff) - 1) + [high_coeff], self.wavelet)

            low.append(low_reconstructed[:len(low_reconstructed) // 2])  # 低频部分
            high.append(high_reconstructed[len(high_reconstructed) // 2:])  # 高频部分

        # 将结果转为张量并返回
        low = torch.tensor(low).float()
        high = torch.tensor(high).float()

        return low, high
