"""
晚期融合模型：决策级别的融合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_predict
import joblib

logger = logging.getLogger(__name__)


class LateFusionModel:
    """晚期融合模型"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化晚期融合模型

        Args:
            config: 模型配置
        """
        self.config = config
        self.stacking_method = config.get('stacking_method', 'logistic')
        self.ensemble_method = config.get('ensemble_method', 'weighted')
        self.cv_folds = config.get('cv_folds', 5)

        # 存储基础模型
        self.tabular_model = None
        self.image_model = None

        # 存储融合模型
        self.stacking_model = None
        self.weights = None

    def fit_base_models(self, tabular_model: Any, image_model: Any):
        """
        设置已训练的基础模型

        Args:
            tabular_model: 表格数据模型
            image_model: 影像数据模型
        """
        self.tabular_model = tabular_model
        self.image_model = image_model

    def train_fusion(self, X_tabular: pd.DataFrame,
                     X_images: pd.DataFrame,
                     y: pd.Series,
                     use_meta_features: bool = True) -> Dict[str, Any]:
        """
        训练融合模型

        Args:
            X_tabular: 表格特征
            X_images: 影像特征或嵌入
            y: 标签
            use_meta_features: 是否使用元特征

        Returns:
            Dict[str, Any]: 训练结果
        """
        if self.tabular_model is None or self.image_model is None:
            raise ValueError("需要先设置基础模型")

        # 获取基础模型的预测
        tabular_probas = self._get_model_predictions(self.tabular_model, X_tabular)
        image_probas = self._get_model_predictions(self.image_model, X_images)

        # 构建融合特征
        fusion_features = self._build_fusion_features(
            tabular_probas, image_probas, X_tabular, X_images, use_meta_features
        )

        # 训练融合模型
        fusion_results = self._train_stacking_model(fusion_features, y)

        # 计算权重（用于加权平均）
        self.weights = self._calculate_weights(tabular_probas, image_probas, y)

        training_results = {
            'fusion_metrics': fusion_results,
            'individual_metrics': {
                'tabular': self._evaluate_model(tabular_probas, y),
                'image': self._evaluate_model(image_probas, y)
            },
            'weights': self.weights
        }

        return training_results

    def _get_model_predictions(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """获取模型预测概率"""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            elif hasattr(model, 'predict'):
                return model.predict(X)
            else:
                raise ValueError("模型不支持预测")
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return np.zeros(len(X))

    def _build_fusion_features(self, tabular_probas: np.ndarray,
                              image_probas: np.ndarray,
                              X_tabular: pd.DataFrame,
                              X_images: pd.DataFrame,
                              use_meta_features: bool) -> pd.DataFrame:
        """构建融合特征"""
        # 基础预测特征
        fusion_df = pd.DataFrame({
            'tabular_proba': tabular_probas,
            'image_proba': image_probas,
            'proba_diff': np.abs(tabular_probas - image_probas),
            'proba_mean': (tabular_probas + image_probas) / 2,
            'probia_max': np.maximum(tabular_probas, image_probas),
            'probia_min': np.minimum(tabular_probas, image_probas)
        })

        # 交互特征
        fusion_df['tabular_weighted'] = tabular_probas * image_probas
        fusion_df['confidence_score'] = 1 - np.abs(tabular_probas - image_probas)

        if use_meta_features:
            # 添加统计特征
            fusion_df['tabular_mean'] = np.mean(tabular_probas)
            fusion_df['image_mean'] = np.mean(image_probas)
            fusion_df['tabular_std'] = np.std(tabular_probas)
            fusion_df['image_std'] = np.std(image_probas)

        return fusion_df

    def _train_stacking_model(self, fusion_features: pd.DataFrame,
                            y: pd.Series) -> Dict[str, float]:
        """训练堆叠模型"""
        # 选择堆叠方法
        if self.stacking_method == 'logistic':
            self.stacking_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.stacking_method == 'gbm':
            self.stacking_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.stacking_method == 'rf':
            self.stacking_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"不支持的堆叠方法: {self.stacking_method}")

        # 训练模型
        self.stacking_model.fit(fusion_features, y)

        # 评估模型
        fusion_probas = self.stacking_model.predict_proba(fusion_features)[:, 1]
        metrics = self._evaluate_model(fusion_probas, y)

        return metrics

    def _evaluate_model(self, probas: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """评估模型"""
        return {
            'auc': roc_auc_score(y, probas),
            'auprc': average_precision_score(y, probas)
        }

    def _calculate_weights(self, tabular_probas: np.ndarray,
                          image_probas: np.ndarray,
                          y: pd.Series) -> Dict[str, float]:
        """计算模型权重"""
        # 计算单独模型的AUC
        tabular_auc = roc_auc_score(y, tabular_probas)
        image_auc = roc_auc_score(y, image_probas)

        # 基于性能计算权重
        total_auc = tabular_auc + image_auc
        tabular_weight = tabular_auc / total_auc
        image_weight = image_auc / total_auc

        return {
            'tabular': tabular_weight,
            'image': image_weight,
            'method': 'performance_based'
        }

    def predict(self, X_tabular: pd.DataFrame,
                X_images: pd.DataFrame,
                method: str = 'stacking') -> Tuple[np.ndarray, np.ndarray]:
        """
        预测

        Args:
            X_tabular: 表格特征
            X_images: 影像特征
            method: 融合方法 ('stacking', 'weighted', 'voting')

        Returns:
            Tuple[np.ndarray, np.ndarray]: (预测概率, 预测类别)
        """
        if self.tabular_model is None or self.image_model is None:
            raise ValueError("需要先设置基础模型")

        # 获取基础预测
        tabular_probas = self._get_model_predictions(self.tabular_model, X_tabular)
        image_probas = self._get_model_predictions(self.image_model, X_images)

        if method == 'stacking' and self.stacking_model is not None:
            # 堆叠融合
            fusion_features = self._build_fusion_features(
                tabular_probas, image_probas, X_tabular, X_images, use_meta_features=True
            )
            final_probas = self.stacking_model.predict_proba(fusion_features)[:, 1]

        elif method == 'weighted' and self.weights is not None:
            # 加权平均融合
            final_probas = (
                self.weights['tabular'] * tabular_probas +
                self.weights['image'] * image_probas
            )

        elif method == 'voting':
            # 简单投票融合
            final_probas = (tabular_probas + image_probas) / 2

        else:
            raise ValueError(f"不支持的融合方法: {method}")

        final_predictions = (final_probas > 0.5).astype(int)

        return final_probas, final_predictions

    def cross_validate_fusion(self, X_tabular: pd.DataFrame,
                            X_images: pd.DataFrame,
                            y: pd.Series) -> Dict[str, Any]:
        """
        交叉验证融合模型

        Args:
            X_tabular: 表格特征
            X_images: 影像特征
            y: 标签

        Returns:
            Dict[str, Any]: 交叉验证结果
        """
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        cv_results = {
            'fold_results': [],
            'mean_auc': 0.0,
            'std_auc': 0.0,
            'mean_auprc': 0.0,
            'std_auprc': 0.0
        }

        all_probas = []
        all_labels = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tabular, y)):
            logger.info(f"交叉验证第 {fold+1}/{self.cv_folds} 折")

            # 分割数据
            X_tab_train, X_tab_val = X_tabular.iloc[train_idx], X_tabular.iloc[val_idx]
            X_img_train, X_img_val = X_images.iloc[train_idx], X_images.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 训练基础模型（简化版本）
            # 在实际应用中，这里应该使用外部的预训练模型
            tabular_probas_train = self._get_model_predictions(self.tabular_model, X_tab_train)
            image_probas_train = self._get_model_predictions(self.image_model, X_img_train)
            tabular_probas_val = self._get_model_predictions(self.tabular_model, X_tab_val)
            image_probas_val = self._get_model_predictions(self.image_model, X_img_val)

            # 训练融合模型
            fusion_features_train = self._build_fusion_features(
                tabular_probas_train, image_probas_train, X_tab_train, X_img_train, True
            )
            fusion_features_val = self._build_fusion_features(
                tabular_probas_val, image_probas_val, X_tab_val, X_img_val, True
            )

            # 临时融合模型
            temp_fusion = LogisticRegression(random_state=42, max_iter=1000)
            temp_fusion.fit(fusion_features_train, y_train)

            # 预测和评估
            fold_probas = temp_fusion.predict_proba(fusion_features_val)[:, 1]
            fold_auc = roc_auc_score(y_val, fold_probas)
            fold_auprc = average_precision_score(y_val, fold_probas)

            fold_result = {
                'fold': fold + 1,
                'auc': fold_auc,
                'auprc': fold_auprc
            }
            cv_results['fold_results'].append(fold_result)

            all_probas.extend(fold_probas)
            all_labels.extend(y_val)

        # 计算总体指标
        cv_results['mean_auc'] = np.mean([r['auc'] for r in cv_results['fold_results']])
        cv_results['std_auc'] = np.std([r['auc'] for r in cv_results['fold_results']])
        cv_results['mean_auprc'] = np.mean([r['auprc'] for r in cv_results['fold_results']])
        cv_results['std_auprc'] = np.std([r['auprc'] for r in cv_results['fold_results']])

        logger.info(f"交叉验证完成: AUC={cv_results['mean_auc']:.4f}±{cv_results['std_auc']:.4f}")

        return cv_results

    def get_feature_importance(self) -> Dict[str, Any]:
        """获取特征重要性"""
        importance_results = {}

        if self.stacking_model is not None:
            if hasattr(self.stacking_model, 'coef_'):
                # 逻辑回归
                feature_names = ['tabular_proba', 'image_proba', 'proba_diff',
                               'proba_mean', 'probia_max', 'probia_min',
                               'tabular_weighted', 'confidence_score']
                importance_results['stacking'] = dict(zip(
                    feature_names[:len(self.stacking_model.coef_[0])],
                    self.stacking_model.coef_[0]
                ))
            elif hasattr(self.stacking_model, 'feature_importances_'):
                # 树模型
                importance_results['stacking'] = self.stacking_model.feature_importances_

        importance_results['weights'] = self.weights

        return importance_results

    def save_fusion_model(self, save_path: str):
        """保存融合模型"""
        model_data = {
            'stacking_model': self.stacking_model,
            'weights': self.weights,
            'config': self.config
        }
        joblib.dump(model_data, save_path)
        logger.info(f"融合模型已保存: {save_path}")

    def load_fusion_model(self, load_path: str):
        """加载融合模型"""
        model_data = joblib.load(load_path)
        self.stacking_model = model_data['stacking_model']
        self.weights = model_data['weights']
        self.config = model_data['config']
        logger.info(f"融合模型已加载: {load_path}")