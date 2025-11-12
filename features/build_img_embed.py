"""
影像嵌入构建模块：从CTA影像中提取深度特征嵌入
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from PIL import Image
import SimpleITK as sitk
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CTADataset(Dataset):
    """CTA影像数据集"""

    def __init__(self, image_paths: List[str], transform=None):
        """
        初始化数据集

        Args:
            image_paths: 影像路径列表
            transform: 数据变换
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """获取单个影像样本"""
        image_path = self.image_paths[idx]

        try:
            # 加载3D影像
            image = self._load_ct_image(image_path)

            # 转换为2.5D表示
            image_2_5d = self._convert_to_2_5d(image)

            if self.transform:
                image_2_5d = self.transform(image_2_5d)

            return image_2_5d, idx

        except Exception as e:
            logger.error(f"加载影像失败 {image_path}: {e}")
            # 返回零张量
            return torch.zeros(3, 224, 224), idx

    def _load_ct_image(self, image_path: str) -> np.ndarray:
        """加载CT影像"""
        # 使用SimpleITK加载DICOM或NIfTI
        if image_path.endswith('.dcm'):
            image = sitk.ReadImage(image_path)
        else:
            image = sitk.ReadImage(image_path)

        # 转换为numpy数组
        array = sitk.GetArrayFromImage(image)  # Shape: (Z, Y, X)

        # 标准化到[0, 1]
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)

        return array

    def _convert_to_2_5d(self, image_3d: np.ndarray) -> np.ndarray:
        """将3D影像转换为2.5D表示"""
        # 选择关键切片
        z_slices = image_3d.shape[0]
        if z_slices >= 3:
            # 选择前、中、后三个切片
            slice_indices = [
                z_slices // 4,      # 前25%
                z_slices // 2,      # 中间50%
                3 * z_slices // 4   # 后75%
            ]
        else:
            # 如果切片数不足3，重复使用
            slice_indices = [0, 0, 0] if z_slices == 1 else [0, 1, 1]

        # 提取切片并调整大小
        selected_slices = []
        target_size = (224, 224)

        for idx in slice_indices:
            slice_img = image_3d[idx]
            # 调整大小
            from skimage.transform import resize
            slice_resized = resize(slice_img, target_size, preserve_range=True)
            selected_slices.append(slice_resized)

        # 堆叠为3通道图像
        image_2_5d = np.stack(selected_slices, axis=0)  # Shape: (3, 224, 224)

        return image_2_5d.astype(np.float32)


class ImageEncoder(nn.Module):
    """影像编码器基类"""

    def __init__(self, backbone: str = 'resnet18', pretrained: bool = True, embed_dim: int = 256):
        """
        初始化编码器

        Args:
            backbone: 骨干网络名称
            pretrained: 是否使用预训练权重
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = embed_dim

        # 构建骨干网络
        if backbone.startswith('resnet'):
            self.backbone = self._build_resnet_backbone(backbone, pretrained)
        elif backbone.startswith('efficientnet'):
            self.backbone = self._build_efficientnet_backbone(backbone, pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 获取特征维度
        if backbone.startswith('resnet'):
            feature_dim = self._get_resnet_feature_dim(backbone)
        else:
            feature_dim = 1280  # EfficientNet-B0

        # 构建投影头
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim)
        )

    def _build_resnet_backbone(self, variant: str, pretrained: bool) -> nn.Module:
        """构建ResNet骨干网络"""
        from torchvision import models

        if variant == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
        elif variant == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
        elif variant == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        # 移除最后的分类层
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        return backbone

    def _build_efficientnet_backbone(self, variant: str, pretrained: bool) -> nn.Module:
        """构建EfficientNet骨干网络"""
        # 这里可以集成efficientnet_pytorch库
        # 暂时使用简单的卷积网络
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def _get_resnet_feature_dim(self, variant: str) -> int:
        """获取ResNet特征维度"""
        dim_mapping = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048
        }
        return dim_mapping.get(variant, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        return embeddings


class ImageEmbeddingBuilder:
    """影像嵌入构建器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化构建器

        Args:
            config: 配置信息
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = config.get('batch_size', 8)
        self.backbone = config.get('backbone', 'resnet18')
        self.embed_dim = config.get('embed_dim', 256)
        self.pretrained = config.get('pretrained', True)

        # 初始化模型
        self.encoder = ImageEncoder(
            backbone=self.backbone,
            pretrained=self.pretrained,
            embed_dim=self.embed_dim
        ).to(self.device)

        # 数据变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def build_embeddings(self, imaging_meta_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建影像嵌入

        Args:
            imaging_meta_df: 影像元数据表

        Returns:
            pd.DataFrame: 包含嵌入的结果表
        """
        logger.info("开始构建影像嵌入")

        # 1. 筛选有影像的患者
        img_available_df = imaging_meta_df[
            imaging_meta_df['img__cta_preop_available'] == 1
        ].copy()

        if len(img_available_df) == 0:
            logger.warning("没有可用的CTA影像数据")
            return imaging_meta_df.copy()

        # 2. 准备影像路径
        image_paths = self._prepare_image_paths(img_available_df)

        # 3. 创建数据集和数据加载器
        dataset = CTADataset(image_paths, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 4. 提取嵌入
        embeddings = self._extract_embeddings(dataloader)

        # 5. 创建嵌入特征名
        embed_cols = [f'img__embed_{i}' for i in range(self.embed_dim)]

        # 6. 构建结果数据框
        result_df = img_available_df.copy()
        embed_df = pd.DataFrame(embeddings, columns=embed_cols)
        result_df = pd.concat([result_df.reset_index(drop=True), embed_df], axis=1)

        logger.info(f"影像嵌入构建完成: {len(result_df)} 患者, {self.embed_dim} 维嵌入")
        return result_df

    def _prepare_image_paths(self, img_df: pd.DataFrame) -> List[str]:
        """准备影像路径列表"""
        image_paths = []

        for _, row in img_df.iterrows():
            if 'img__path' in row and pd.notna(row['img__path']):
                image_paths.append(row['img__path'])
            else:
                # 如果没有路径，尝试从默认路径构造
                pid = row['pid']
                default_path = f"data/cta_images/{pid}.nii.gz"
                image_paths.append(default_path)

        return image_paths

    def _extract_embeddings(self, dataloader: DataLoader) -> np.ndarray:
        """提取影像嵌入"""
        self.encoder.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch_idx, (images, indices) in enumerate(dataloader):
                images = images.to(self.device)

                # 提取嵌入
                embeddings = self.encoder(images)
                embeddings = embeddings.cpu().numpy()

                all_embeddings.append(embeddings)

                if batch_idx % 10 == 0:
                    logger.info(f"已处理 {batch_idx * len(images)} / {len(dataloader.dataset)} 影像")

        # 合并所有嵌入
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings

    def save_encoder(self, save_path: str):
        """保存编码器"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'config': self.config
        }, save_path)
        logger.info(f"影像编码器已保存: {save_path}")

    def load_encoder(self, load_path: str):
        """加载编码器"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        logger.info(f"影像编码器已加载: {load_path}")

    def compute_embeddings_for_single_image(self, image_path: str) -> np.ndarray:
        """为单张影像计算嵌入"""
        # 创建单样本数据集
        dataset = CTADataset([image_path], transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # 提取嵌入
        embeddings = self._extract_embeddings(dataloader)
        return embeddings[0]  # 返回单个嵌入向量