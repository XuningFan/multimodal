"""
中期融合模型：交互层级的跨注意力融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from .fusion_early import TabularMLP, MultimodalDataset

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """多头注意力模块"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        初始化多头注意力

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            mask: 注意力掩码 [batch_size, seq_len]

        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.size()

        # 计算Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)
        V = self.value(x)

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, head_dim]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, head_dim]

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # 输出投影
        output = self.out_proj(attn_output)

        return output


class CrossModalAttention(nn.Module):
    """跨模态注意力模块"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        初始化跨模态注意力

        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 自注意力层
        self.tabular_self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.image_self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # 跨模态注意力
        self.tabular_to_image_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.image_to_tabular_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

        # Feed-forward网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, tabular_features: torch.Tensor,
                image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            tabular_features: 表格特征 [batch_size, embed_dim]
            image_features: 影像特征 [batch_size, embed_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (增强的表格特征, 增强的影像特征)
        """
        # 添加序列维度
        tabular = tabular_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        image = image_features.unsqueeze(1)      # [batch_size, 1, embed_dim]

        # 表格特征的自注意力
        tabular_self = self.tabular_self_attn(tabular)
        tabular = self.norm1(tabular + tabular_self)

        # 影像特征的自注意力
        image_self = self.image_self_attn(image)
        image = self.norm2(image + image_self)

        # 跨模态注意力：表格关注影像
        tabular_cross = self.tabular_to_image_attn(tabular, mask=torch.ones_like(image))
        tabular = self.norm3(tabular + tabular_cross)

        # 跨模态注意力：影像关注表格
        image_cross = self.image_to_tabular_attn(image, mask=torch.ones_like(tabular))
        image = self.norm4(image + image_cross)

        # Feed-forward网络
        tabular_ffn = self.ffn(tabular)
        tabular = tabular + tabular_ffn

        image_ffn = self.ffn(image)
        image = image + image_ffn

        # 移除序列维度
        return tabular.squeeze(1), image.squeeze(1)


class IntermediateFusionModel(nn.Module):
    """中期融合模型"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化中期融合模型

        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.tabular_input_dim = config.get('tabular_input_dim', 100)
        self.embed_dim = config.get('embed_dim', 256)
        self.num_attention_layers = config.get('num_attention_layers', 2)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.2)

        # 表格数据编码器
        self.tabular_encoder = TabularMLP(
            input_dim=self.tabular_input_dim,
            layers=[128, 64],
            dropout=self.dropout
        )

        # 影像编码器（从外部传入）
        self.image_encoder = None

        # 跨模态注意力层
        self.attention_layers = nn.ModuleList([
            CrossModalAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            ) for _ in range(self.num_attention_layers)
        ])

        # 融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def set_image_encoder(self, image_encoder: nn.Module):
        """设置影像编码器"""
        self.image_encoder = image_encoder

    def forward(self, tabular_data: torch.Tensor,
                image_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            tabular_data: 表格数据 [batch_size, tabular_dim]
            image_data: 影像数据 [batch_size, 3, H, W] 或影像嵌入 [batch_size, embed_dim]

        Returns:
            torch.Tensor: 预测概率 [batch_size, 1]
        """
        # 编码表格数据
        tabular_features = self.tabular_encoder(tabular_data)

        if image_data is not None:
            # 编码影像数据
            if image_data.dim() == 4:  # 原始影像数据
                if self.image_encoder is None:
                    raise ValueError("需要设置影像编码器")
                image_features = self.image_encoder(image_data)
            else:  # 预计算的影像嵌入
                image_features = image_data

            # 通过跨模态注意力层
            for attention_layer in self.attention_layers:
                tabular_features, image_features = attention_layer(
                    tabular_features, image_features
                )

            # 融合特征
            fused_features = torch.cat([tabular_features, image_features], dim=1)
        else:
            # 仅表格数据
            fused_features = tabular_features

        # 分类预测
        output = self.fusion_classifier(fused_features)

        return output

    def get_attention_weights(self, tabular_data: torch.Tensor,
                             image_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取注意力权重（用于可视化）

        Args:
            tabular_data: 表格数据
            image_data: 影像数据

        Returns:
            Dict[str, torch.Tensor]: 注意力权重
        """
        # 编码数据
        tabular_features = self.tabular_encoder(tabular_data)
        if image_data.dim() == 4:
            image_features = self.image_encoder(image_data)
        else:
            image_features = image_data

        attention_weights = {}

        # 通过注意力层并记录权重
        tabular = tabular_features.unsqueeze(1)
        image = image_features.unsqueeze(1)

        for i, attention_layer in enumerate(self.attention_layers):
            # 这里需要修改CrossModalAttention以返回注意力权重
            # 暂时返回输入特征
            tabular, image = attention_layer(tabular, image)
            attention_weights[f'layer_{i}'] = {
                'tabular_features': tabular,
                'image_features': image
            }

        return attention_weights