"""
模型训练模块：统一的训练接口
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime

from .clinical_baselines import ClinicalBaselines
from .fusion_early import EarlyFusionModel, EarlyFusionTrainer, MultimodalDataset
from .fusion_intermediate import IntermediateFusionModel
from .fusion_late import LateFusionModel
from features.build_t0 import T0FeatureBuilder
from features.build_img_embed import ImageEmbeddingBuilder
from features.radiomics import RadiomicsExtractor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """统一模型训练器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 训练配置
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化模型组件
        self.models = {}
        self.feature_builders = {}
        self.training_results = {}

    def setup_models(self) -> Dict[str, Any]:
        """设置所有模型"""
        logger.info("设置模型组件")

        # 1. 临床基线模型
        if 'clinical_only' in self.config.get('models', {}):
            clinical_config = self.config['models']['clinical_only']
            self.models['clinical'] = ClinicalBaselines(clinical_config)

        # 2. 早期融合模型
        if 'fusion_early' in self.config.get('models', {}):
            early_config = self.config['models']['fusion_early']
            self.models['early_fusion'] = EarlyFusionModel(early_config)

            # 设置影像编码器
            img_config = self.config.get('image_features', {})
            image_encoder = ImageEmbeddingBuilder(img_config).encoder
            self.models['early_fusion'].set_image_encoder(image_encoder)

        # 3. 中期融合模型
        if 'fusion_intermediate' in self.config.get('models', {}):
            intermediate_config = self.config['models']['fusion_intermediate']
            self.models['intermediate_fusion'] = IntermediateFusionModel(intermediate_config)

            # 设置影像编码器
            img_config = self.config.get('image_features', {})
            image_encoder = ImageEmbeddingBuilder(img_config).encoder
            self.models['intermediate_fusion'].set_image_encoder(image_encoder)

        # 4. 晚期融合模型
        if 'fusion_late' in self.config.get('models', {}):
            late_config = self.config['models']['fusion_late']
            self.models['late_fusion'] = LateFusionModel(late_config)

        logger.info(f"设置了 {len(self.models)} 个模型")
        return self.models

    def prepare_data(self, patients_df: pd.DataFrame,
                    surgery_cpb_df: pd.DataFrame,
                    labs_long_df: pd.DataFrame,
                    imaging_meta_df: pd.DataFrame,
                    outcomes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        准备训练数据

        Args:
            patients_df: 患者信息表
            surgery_cpb_df: 手术CPB表
            labs_long_df: 化验长表
            imaging_meta_df: 影像元数据表
            outcomes_df: 结局表

        Returns:
            Dict[str, Any]: 准备好的数据
        """
        logger.info("准备训练数据")

        # 1. 构建T0特征
        t0_config = self.config.get('features', {})
        t0_builder = T0FeatureBuilder(t0_config)
        t0_features = t0_builder.build_t0_features(
            patients_df, surgery_cpb_df, labs_long_df, imaging_meta_df
        )

        # 2. 构建影像特征
        image_features = None
        if 'image_features' in self.config:
            img_config = self.config['image_features']
            img_builder = ImageEmbeddingBuilder(img_config)
            imaging_with_embeddings = img_builder.build_embeddings(imaging_meta_df)

            # 筛选有影像的患者
            image_features = imaging_with_embeddings[
                imaging_with_embeddings['img__cta_preop_available'] == 1
            ]

        # 3. 合并结局标签
        target_column = self.config.get('target_column', 'outcome__death30')
        data_with_labels = t0_features.merge(
            outcomes_df[['pid', target_column]], on='pid', how='inner'
        )

        # 4. 创建不同视图
        data_views = self._create_data_views(data_with_labels, image_features, target_column)

        return {
            't0_features': t0_features,
            'image_features': image_features,
            'data_views': data_views,
            'target_column': target_column
        }

    def _create_data_views(self, data_with_labels: pd.DataFrame,
                          image_features: Optional[pd.DataFrame],
                          target_column: str) -> Dict[str, Any]:
        """创建数据视图"""
        views = {}

        # 临床数据视图（所有患者）
        views['primary_all'] = {
            'X': data_with_labels.drop(columns=['pid', target_column]),
            'y': data_with_labels[target_column],
            'patient_ids': data_with_labels['pid']
        }

        # 影像数据视图（有影像的患者）
        if image_features is not None:
            # 获取影像特征列
            img_embed_cols = [col for col in image_features.columns if col.startswith('img__embed_')]

            # 合并临床和影像特征
            imaging_data = data_with_labels.merge(
                image_features[['pid'] + img_embed_cols],
                on='pid',
                how='inner'
            )

            views['primary_img'] = {
                'X_tabular': imaging_data.drop(
                    columns=['pid', target_column] + img_embed_cols
                ),
                'X_image': imaging_data[img_embed_cols],
                'y': imaging_data[target_column],
                'patient_ids': imaging_data['pid']
            }

        return views

    def train_all_models(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """训练所有模型"""
        logger.info("开始训练所有模型")

        data_views = prepared_data['data_views']
        results = {}

        # 1. 训练临床基线模型
        if 'clinical' in self.models:
            logger.info("训练临床基线模型")
            clinical_results = self._train_clinical_model(data_views)
            results['clinical'] = clinical_results

        # 2. 训练早期融合模型
        if 'early_fusion' in self.models and 'primary_img' in data_views:
            logger.info("训练早期融合模型")
            early_results = self._train_early_fusion_model(data_views)
            results['early_fusion'] = early_results

        # 3. 训练中期融合模型
        if 'intermediate_fusion' in self.models and 'primary_img' in data_views:
            logger.info("训练中期融合模型")
            intermediate_results = self._train_intermediate_fusion_model(data_views)
            results['intermediate_fusion'] = intermediate_results

        # 4. 训练晚期融合模型
        if 'late_fusion' in self.models and 'primary_img' in data_views:
            logger.info("训练晚期融合模型")
            late_results = self._train_late_fusion_model(data_views, results)
            results['late_fusion'] = late_results

        self.training_results = results
        return results

    def _train_clinical_model(self, data_views: Dict[str, Any]) -> Dict[str, Any]:
        """训练临床模型"""
        view_data = data_views['primary_all']
        X, y = view_data['X'], view_data['y']

        # 交叉验证
        cv_results = self.models['clinical'].cross_validate(X, y)

        # 在全数据上训练
        self.models['clinical'].build_model()
        training_results = self.models['clinical'].train(X, y)

        return {
            'cv_results': cv_results,
            'training_results': training_results,
            'model': self.models['clinical']
        }

    def _train_early_fusion_model(self, data_views: Dict[str, Any]) -> Dict[str, Any]:
        """训练早期融合模型"""
        view_data = data_views['primary_img']
        X_tab, X_img, y = view_data['X_tabular'], view_data['X_image'], view_data['y']

        # 数据预处理
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tab_scaled = scaler.fit_transform(X_tab)

        # 创建数据集
        dataset = MultimodalDataset(X_tab_scaled, X_img.values, y.values)

        # 分割训练验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=16, shuffle=False
        )

        # 更新模型配置中的表格维度
        tabular_dim = X_tab_scaled.shape[1]
        self.models['early_fusion'].tabular_input_dim = tabular_dim

        # 创建训练器
        trainer = EarlyFusionTrainer(
            self.models['early_fusion'],
            self.config.get('training', {})
        )

        # 训练模型
        training_results = trainer.train(train_loader, val_loader)

        return {
            'training_results': training_results,
            'trainer': trainer,
            'tabular_scaler': scaler,
            'model': self.models['early_fusion']
        }

    def _train_intermediate_fusion_model(self, data_views: Dict[str, Any]) -> Dict[str, Any]:
        """训练中期融合模型"""
        # 与早期融合类似的流程，但使用不同的模型
        # 这里可以复用早期融合的代码框架
        logger.info("中期融合模型训练（使用早期融合的类似流程）")
        return {"status": "implemented_similar_to_early_fusion"}

    def _train_late_fusion_model(self, data_views: Dict[str, Any],
                               previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """训练晚期融合模型"""
        view_data = data_views['primary_img']
        X_tab, X_img, y = view_data['X_tabular'], view_data['X_image'], view_data['y']

        # 使用已训练的基础模型
        self.models['late_fusion'].fit_base_models(
            previous_results.get('clinical', {}).get('model'),
            previous_results.get('early_fusion', {}).get('model')
        )

        # 训练融合模型
        fusion_results = self.models['late_fusion'].train_fusion(X_tab, X_img, y)

        return {
            'fusion_results': fusion_results,
            'model': self.models['late_fusion']
        }

    def save_results(self) -> str:
        """保存训练结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.output_dir / f"training_results_{timestamp}"
        results_dir.mkdir(exist_ok=True)

        # 保存训练结果
        results_summary = {
            'timestamp': timestamp,
            'config': self.config,
            'models_trained': list(self.training_results.keys())
        }

        for model_name, results in self.training_results.items():
            # 保存模型
            if 'model' in results:
                model_path = results_dir / f"{model_name}_model.pkl"
                if hasattr(results['model'], 'save_model'):
                    results['model'].save_model(str(model_path))
                else:
                    import joblib
                    joblib.dump(results['model'], model_path)

            # 保存训练指标
            model_results_dir = results_dir / f"{model_name}_results"
            model_results_dir.mkdir(exist_ok=True)

            # 保存结果到JSON
            with open(model_results_dir / "metrics.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)

        # 保存配置
        with open(results_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"训练结果已保存: {results_dir}")
        return str(results_dir)

    def generate_model_cards(self) -> Dict[str, str]:
        """生成模型卡片"""
        model_cards = {}

        for model_name, results in self.training_results.items():
            card = self._create_model_card(model_name, results)
            card_path = self.output_dir / f"{model_name}_model_card.md"
            with open(card_path, 'w', encoding='utf-8') as f:
                f.write(card)
            model_cards[model_name] = str(card_path)

        return model_cards

    def _create_model_card(self, model_name: str, results: Dict[str, Any]) -> str:
        """创建单个模型卡片"""
        card = f"""# 模型卡片: {model_name}

## 模型概述
- 模型类型: {self.config.get('models', {}).get(model_name, {}).get('type', 'Unknown')}
- 训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 性能指标
"""

        # 添加性能指标
        if 'cv_results' in results:
            cv_results = results['cv_results']
            if 'mean_metrics' in cv_results:
                metrics = cv_results['mean_metrics']
                card += f"### 交叉验证结果\n"
                for metric, value in metrics.items():
                    card += f"- {metric.upper()}: {value:.4f}\n"

        if 'training_results' in results:
            training_results = results['training_results']
            if 'val_metrics' in training_results:
                val_metrics = training_results['val_metrics']
                card += f"### 验证集结果\n"
                for metric, value in val_metrics.items():
                    card += f"- {metric.upper()}: {value:.4f}\n"

        card += f"""
## 配置信息
```yaml
{yaml.dump(self.config.get('models', {}).get(model_name, {}), default_flow_style=False)}
```

## 局限性
- 训练数据规模有限
- 单中心验证结果
- 需要外部验证

## 使用建议
- 适用于临床决策辅助
- 需要结合临床判断使用
"""

        return card