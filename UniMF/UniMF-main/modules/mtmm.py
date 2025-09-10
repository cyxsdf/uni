import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder  # 复用原Transformer


class MultiScaleTCN(nn.Module):
    """多尺度时序卷积网络：捕捉局部短时依赖"""

    def __init__(self, input_dim, hidden_dim, kernel_sizes=[3, 5, 7], dropout=0.2):
        super(MultiScaleTCN, self).__init__()
        self.kernel_sizes = kernel_sizes
        # 多尺度卷积分支
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    kernel_size=k,
                    padding=k // 2  # 保持序列长度不变
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])
        # 跨分支融合
        self.fusion = nn.Linear(hidden_dim * len(kernel_sizes), input_dim)

    def forward(self, x):
        """
        x: (seq_len, batch_size, input_dim) → 转换为 (batch_size, input_dim, seq_len) 用于1D卷积
        return: (seq_len, batch_size, input_dim) 局部特征增强后的序列
        """
        x = x.permute(1, 2, 0)  # (B, D, T)

        # 多尺度卷积
        conv_outs = []
        for conv in self.conv_branches:
            out = conv(x)  # (B, H, T)
            conv_outs.append(out)

        # 拼接多尺度特征并融合
        conv_outs = torch.cat(conv_outs, dim=1)  # (B, H*K, T)
        fusion_out = self.fusion(conv_outs.permute(0, 2, 1))  # (B, T, D)
        fusion_out = fusion_out.permute(1, 0, 2)  # (T, B, D)

        # 残差连接
        return fusion_out + x.permute(2, 0, 1)  # (T, B, D)


class MTMM(nn.Module):
    """多粒度时序建模模块：MS-TCN + Transformer"""

    def __init__(self, embed_dim, num_heads, layers, lens, modalities,
                 kernel_sizes=[3, 5, 7], attn_dropout=0.1, relu_dropout=0.1,
                 missing=None):  # 新增missing参数
        super(MTMM, self).__init__()
        # 保存缺失模态信息
        self.missing = missing
        self.modalities = modalities

        # 多尺度TCN（局部建模）
        self.ms_tcn = MultiScaleTCN(
            input_dim=embed_dim,
            hidden_dim=embed_dim // 2,
            kernel_sizes=kernel_sizes,
            dropout=relu_dropout
        )
        # Transformer（全局建模）
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            lens=lens,
            modalities=modalities,
            missing=self.missing,  # 将缺失模态信息传递给Transformer
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout
        )

    def forward(self, x):
        """
        x: (seq_len, batch_size, embed_dim) 输入时序特征
        return: (seq_len, batch_size, embed_dim) 多粒度建模后的特征
        """
        # 1. 局部时序建模（MS-TCN）
        local_feat = self.ms_tcn(x)

        # 2. 全局长时序建模（Transformer）- 已包含缺失模态处理逻辑
        global_feat = self.transformer(local_feat)

        return global_feat
