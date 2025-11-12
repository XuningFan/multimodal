"""
早期融合模型：特征级别的多模态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TabularMLP(nn.Module):
    """表格数据MLP编码器"""

    def __init__(self, input_dim: int, layers: List[int], dropout: float = 0.2):
        """
        初始化表格MLP

        Args:
            input_dim: 输入维度
            layers: 隐藏层维度列表
            dropout: Dropout比率
        """
        super().__init__()
        self.input_dim = input_dim

        # 构建网络层
        layers_list = []
        prev_dim = input_dim

        for layer_dim in layers:
            layers_list.extend([
                nn.Linear(prev_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = layer_dim

        # 最终输出层（降维到融合维度）
        layers_list.append(nn.Linear(prev_dim, 256))

        self.mlp = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.mlp(x)


class EarlyFusionModel(nn.Module):
    """早期融合模型"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化早期融合模型

        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
        self.tabular_input_dim = config.get('tabular_input_dim', 100)
        self.tabular_layers = config.get('tabular_layers', [128, 64])
        self.dropout = config.get('dropout', 0.2)
        self.embed_dim = config.get('embed_dim', 256)

        # 表格数据编码器
        self.tabular_encoder = TabularMLP(
            input_dim=self.tabular_input_dim,
            layers=self.tabular_layers,
            dropout=self.dropout
        )

        # 影像编码器（从外部传入）
        self.image_encoder = None

        # 融合层
        fusion_input_dim = self.embed_dim * 2  # 表格 + 影像
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
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
        tabular_embed = self.tabular_encoder(tabular_data)

        if image_data is not None:
            # 编码影像数据
            if image_data.dim() == 4:  # 原始影像数据
                if self.image_encoder is None:
                    raise ValueError("需要设置影像编码器")
                image_embed = self.image_encoder(image_data)
            else:  # 预计算的影像嵌入
                image_embed = image_data

            # 早期融合：特征连接
            fused_features = torch.cat([tabular_embed, image_embed], dim=1)
        else:
            # 仅表格数据
            fused_features = tabular_embed

        # 通过融合层得到预测
        output = self.fusion_layers(fused_features)

        return output

    def get_embeddings(self, tabular_data: torch.Tensor,
                      image_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取融合嵌入

        Args:
            tabular_data: 表格数据
            image_data: 影像数据

        Returns:
            torch.Tensor: 融合嵌入
        """
        tabular_embed = self.tabular_encoder(tabular_data)

        if image_data is not None:
            if image_data.dim() == 4:
                image_embed = self.image_encoder(image_data)
            else:
                image_embed = image_data

            fused_embed = torch.cat([tabular_embed, image_embed], dim=1)
            return fused_embed
        else:
            return tabular_embed


class MultimodalDataset(Dataset):
    """多模态数据集"""

    def __init__(self, tabular_features: np.ndarray,
                 image_embeddings: Optional[np.ndarray] = None,
                 labels: Optional[np.ndarray] = None):
        """
        初始化数据集

        Args:
            tabular_features: 表格特征 [N, tabular_dim]
            image_embeddings: 影像嵌入 [N, embed_dim]
            labels: 标签 [N]
        """
        self.tabular_features = torch.FloatTensor(tabular_features)
        self.image_embeddings = torch.FloatTensor(image_embeddings) if image_embeddings is not None else None
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.tabular_features)

    def __getitem__(self, idx):
        """获取单个样本"""
        item = {
            'tabular': self.tabular_features[idx],
            'image': self.image_embeddings[idx] if self.image_embeddings is not None else None,
            'index': idx
        }

        if self.labels is not None:
            item['label'] = self.labels[idx]

        return item


class EarlyFusionTrainer:
    """早期融合模型训练器"""

    def __init__(self, model: EarlyFusionModel, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            model: 早期融合模型
            config: 训练配置
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将模型移到设备
        self.model.to(self.device)

        # 设置优化器
        self.optimizer = self._setup_optimizer()

        # 设置损失函数
        self.criterion = self._setup_criterion()

        # 学习率调度器
        self.scheduler = self._setup_scheduler()

        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """设置优化器"""
        optimizer_type = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _setup_criterion(self) -> nn.Module:
        """设置损失函数"""
        pos_weight = self.config.get('pos_weight', 1.0)

        return nn.BCELoss()

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """设置学习率调度器"""
        scheduler_type = self.config.get('scheduler')

        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('max_epochs', 100)
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.get('patience', 10),
                factor=0.5
            )
        else:
            return None

    def train(self, train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              max_epochs: int = 100,
              patience: int = 20) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            max_epochs: 最大训练轮数
            patience: 早停耐心值

        Returns:
            Dict[str, Any]: 训练结果
        """
        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(max_epochs):
            # 训练阶段
            train_loss, train_auc = self._train_epoch(train_loader)

            # 验证阶段
            val_loss, val_auc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_auc = self._validate_epoch(val_loader)

            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_auc'].append(train_auc)
            if val_loader is not None:
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_auc'].append(val_auc)

            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 打印进度
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}")
                if val_loader is not None:
                    logger.info(f"         Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")

            # 早停检查
            if val_loader is not None and val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"早停于第 {epoch} 轮，最佳验证AUC: {best_val_auc:.4f}")
                self.model.load_state_dict(best_model_state)
                break

        training_results = {
            'train_history': self.train_history,
            'best_val_auc': best_val_auc,
            'total_epochs': epoch + 1
        }

        return training_results

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for batch in train_loader:
            tabular = batch['tabular'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            images = batch['image'].to(self.device) if batch['image'] is not None else None

            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(tabular, images)
            loss = self.criterion(predictions, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        # 计算AUC
        from sklearn.metrics import roc_auc_score
        train_auc = roc_auc_score(all_labels, all_predictions)

        return total_loss / len(train_loader), train_auc

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                tabular = batch['tabular'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)
                images = batch['image'].to(self.device) if batch['image'] is not None else None

                predictions = self.model(tabular, images)
                loss = self.criterion(predictions, labels)

                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_predictions)

        return total_loss / len(val_loader), val_auc

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测

        Args:
            data_loader: 数据加载器

        Returns:
            Tuple[np.ndarray, np.ndarray]: (预测概率, 预测类别)
        """
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                tabular = batch['tabular'].to(self.device)
                images = batch['image'].to(self.device) if batch['image'] is not None else None

                predictions = self.model(tabular, images)
                all_predictions.extend(predictions.cpu().numpy())

        predictions = np.array(all_predictions).flatten()
        classes = (predictions > 0.5).astype(int)

        return predictions, classes

    def save_model(self, save_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_history': self.train_history
        }, save_path)
        logger.info(f"模型已保存: {save_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', {})
        logger.info(f"模型已加载: {load_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_config': self.config,
            'device': str(self.device)
        }