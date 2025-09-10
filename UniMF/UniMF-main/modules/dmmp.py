# UniMF/modules/dmmp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DynamicMemoryPool(nn.Module):
    """动态多模态记忆池：存储和检索典型情感模式"""

    def __init__(self, embed_dim, memory_size=1000, top_k=5):
        super(DynamicMemoryPool, self).__init__()
        self.embed_dim = embed_dim  # 特征维度
        self.memory_size = memory_size  # 记忆容量
        self.top_k = top_k  # 检索相似记忆的数量

        # 初始化记忆库（随机正态分布）
        self.memory = nn.Parameter(torch.randn(memory_size, embed_dim))
        # 记忆的情感标签（0: 负, 1: 中, 2: 正，可扩展）
        self.memory_labels = nn.Parameter(torch.randint(0, 3, (memory_size,)), requires_grad=False)
        # 记忆的访问频率（用于动态更新）
        self.access_freq = nn.Parameter(torch.zeros(memory_size), requires_grad=False)

    def retrieve(self, query, label=None):
        """
        检索与查询特征相似的记忆
        query: (batch_size, seq_len, embed_dim)
        label: 情感标签（可选，用于过滤记忆）
        return: (batch_size, seq_len, top_k, embed_dim) 相似记忆特征
        """
        # 计算查询与记忆的相似度（余弦距离）
        query_norm = F.normalize(query, dim=-1)  # (B, T, D)
        memory_norm = F.normalize(self.memory, dim=-1)  # (M, D)
        similarity = torch.matmul(query_norm, memory_norm.t())  # (B, T, M)

        # 按情感标签过滤（如果指定）
        if label is not None:
            # label: (B, T)，扩展为 (B, T, M)
            label_expand = label.unsqueeze(-1).expand(-1, -1, self.memory_size)
            # 仅保留标签匹配的记忆
            mask = (label_expand == self.memory_labels.unsqueeze(0).unsqueeze(0))
            similarity = similarity.masked_fill(~mask, -1e9)

        # 取top-k相似记忆
        top_k_sim, top_k_idx = torch.topk(similarity, k=self.top_k, dim=-1)  # (B, T, K), (B, T, K)
        # 提取对应的记忆特征
        batch_size, seq_len, _ = query.shape
        top_k_memory = self.memory[top_k_idx]  # (B, T, K, D)

        # 更新访问频率（增加被检索记忆的频率）
        if self.training:
            with torch.no_grad():
                # 展平索引并累加频率
                flat_idx = top_k_idx.view(-1)
                self.access_freq.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

        return top_k_memory, top_k_sim

    def update(self, new_features, new_labels, threshold=0.5):
        """
        动态更新记忆库：添加新的典型模式，替换低频记忆
        new_features: (N, embed_dim) 新的特征
        new_labels: (N,) 新特征的情感标签
        threshold: 新特征与记忆库的最小相似度，低于此值则视为新模式
        """
        if not self.training:
            return

        with torch.no_grad():
            # 计算新特征与现有记忆的相似度
            new_norm = F.normalize(new_features, dim=-1)
            memory_norm = F.normalize(self.memory, dim=-1)
            sim = torch.matmul(new_norm, memory_norm.t())  # (N, M)
            max_sim, _ = sim.max(dim=-1)  # (N,) 每个新特征与记忆库的最大相似度

            # 筛选出与现有记忆差异大的新特征（视为新模式）
            new_mask = (max_sim < threshold)
            new_features = new_features[new_mask]
            new_labels = new_labels[new_mask]
            if len(new_features) == 0:
                return

            # 替换访问频率最低的记忆
            num_replace = min(len(new_features), self.memory_size)
            freq_sorted_idx = torch.argsort(self.access_freq)  # 升序排列（频率最低的在前）
            replace_idx = freq_sorted_idx[:num_replace]

            # 更新记忆库和标签
            self.memory[replace_idx] = new_features[:num_replace]
            self.memory_labels[replace_idx] = new_labels[:num_replace]
            self.access_freq[replace_idx] = 0  # 重置新记忆的访问频率

    def forward(self, x, labels=None):
        """
        x: 多模态融合特征 (batch_size, seq_len, embed_dim)
        labels: 情感标签 (batch_size, seq_len)（可选）
        return: 融合记忆后的特征 (batch_size, seq_len, embed_dim)
        """
        # 检索相似记忆
        top_k_memory, top_k_sim = self.retrieve(x, labels)  # (B, T, K, D), (B, T, K)

        # 加权融合记忆（按相似度权重）
        weights = F.softmax(top_k_sim, dim=-1).unsqueeze(-1)  # (B, T, K, 1)
        memory_feat = (top_k_memory * weights).sum(dim=2)  # (B, T, D)

        # 与输入特征融合
        fused_feat = x + memory_feat  # 残差连接
        return fused_feat