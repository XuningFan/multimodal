"""
临床基线模型：仅使用临床数据的基线模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import shap

logger = logging.getLogger(__name__)


class ClinicalBaselines:
    """临床基线模型集合"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化基线模型

        Args:
            config: 模型配置
        """
        self.config = config
        self.model_type = config.get('model_type', 'lightgbm')
        self.model_params = config.get('params', {})
        self.scaler_type = config.get('scaler', 'standard')
        self.calibration_method = config.get('calibration', 'platt')

        # 初始化模型
        self.model = None
        self.scaler = None
        self.calibrator = None

    def build_model(self) -> Union[lgb.LGBMClassifier, xgb.XGBClassifier, LogisticRegression]:
        """构建模型"""
        if self.model_type == 'lightgbm':
            default_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_estimators': 200,
                'class_weight': 'balanced'
            }
            params = {**default_params, **self.model_params}
            model = lgb.LGBMClassifier(**params)

        elif self.model_type == 'xgboost':
            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'scale_pos_weight': 1,
                'use_label_encoder': False
            }
            params = {**default_params, **self.model_params}
            model = xgb.XGBClassifier(**params)

        elif self.model_type == 'logistic':
            default_params = {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'liblinear',
                'random_state': 42,
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
            params = {**default_params, **self.model_params}
            model = LogisticRegression(**params)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model = model
        logger.info(f"构建模型: {self.model_type}")
        return model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            feature_names: 特征名称列表

        Returns:
            Dict[str, Any]: 训练结果
        """
        # 预处理
        X_train_processed = self._preprocess_features(X_train, fit=True)

        if X_val is not None:
            X_val_processed = self._preprocess_features(X_val, fit=False)
        else:
            X_val_processed = None

        # 构建模型（如果还没有）
        if self.model is None:
            self.build_model()

        # 训练
        if self.model_type in ['lightgbm', 'xgboost'] and X_val_processed is not None:
            # 使用验证集进行早停
            if self.model_type == 'lightgbm':
                self.model.fit(
                    X_train_processed, y_train,
                    eval_set=[(X_val_processed, y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            else:  # xgboost
                self.model.fit(
                    X_train_processed, y_train,
                    eval_set=[(X_val_processed, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
        else:
            # 简单训练
            self.model.fit(X_train_processed, y_train)

        # 校准
        if X_val_processed is not None:
            self._calibrate_model(X_val_processed, y_val)

        # 评估
        train_metrics = self._evaluate_model(X_train_processed, y_train, "训练集")
        val_metrics = {}
        if X_val_processed is not None:
            val_metrics = self._evaluate_model(X_val_processed, y_val, "验证集")

        # 特征重要性
        feature_importance = self._get_feature_importance(feature_names or X_train.columns.tolist())

        training_results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': feature_importance,
            'model_params': self.get_model_params()
        }

        logger.info(f"模型训练完成: {self.model_type}")
        return training_results

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测

        Args:
            X: 特征数据

        Returns:
            Tuple[np.ndarray, np.ndarray]: (预测概率, 预测类别)
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        X_processed = self._preprocess_features(X, fit=False)

        # 预测概率
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_processed)[:, 1]
        else:
            probas = self.model.predict(X_processed)

        # 校准
        if self.calibrator is not None:
            probas = self.calibrator.predict_proba(probas.reshape(-1, 1))[:, 1]

        # 预测类别
        predictions = (probas > 0.5).astype(int)

        return probas, predictions

    def _preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """特征预处理"""
        # 获取数值特征
        numeric_features = X.select_dtypes(include=[np.number]).columns

        if len(numeric_features) == 0:
            return X.values

        X_numeric = X[numeric_features]

        # 标准化
        if fit:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                return X.values

            X_scaled = self.scaler.fit_transform(X_numeric)
        else:
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X_numeric)
            else:
                X_scaled = X_numeric.values

        # 如果还有分类特征，需要额外处理
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_features) > 0:
            # 简单的标签编码
            from sklearn.preprocessing import LabelEncoder
            X_cat = X[categorical_features].apply(
                lambda col: LabelEncoder().fit_transform(col.astype(str))
            )
            return np.hstack([X_scaled, X_cat.values])
        else:
            return X_scaled

    def _calibrate_model(self, X_val: pd.DataFrame, y_val: pd.Series):
        """校准模型"""
        from sklearn.calibration import CalibratedClassifierCV

        # 获取原始预测
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_val)[:, 1]
        else:
            probas = self.model.predict(X_val)

        if self.calibration_method == 'platt':
            self.calibrator = LogisticRegression()
        elif self.calibration_method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            return

        # 拟合校准器
        self.calibrator.fit(probas.reshape(-1, 1), y_val)

    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict[str, float]:
        """评估模型"""
        probas, predictions = self.predict(X)

        metrics = {
            'auc': roc_auc_score(y, probas),
            'auprc': average_precision_score(y, probas),
            'brier': brier_score_loss(y, probas)
        }

        logger.info(f"{dataset_name}评估结果: AUC={metrics['auc']:.4f}, AUPRC={metrics['auprc']:.4f}")

        return metrics

    def _get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """获取特征重要性"""
        if self.model is None:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            logger.warning("模型不支持特征重要性计算")
            return {}

        return dict(zip(feature_names, importances))

    def explain_predictions(self, X: pd.DataFrame, background_samples: Optional[int] = 100) -> Dict[str, Any]:
        """
        使用SHAP解释预测

        Args:
            X: 待解释的特征数据
            background_samples: 背景样本数量

        Returns:
            Dict[str, Any]: SHAP解释结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        try:
            # 处理数据
            X_processed = self._preprocess_features(X, fit=False)

            # 创建背景数据
            if background_samples and len(X_processed) > background_samples:
                background = shap.sample(X_processed, background_samples)
            else:
                background = X_processed

            # 创建SHAP解释器
            if self.model_type in ['lightgbm', 'xgboost']:
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.KernelExplainer(self.model.predict_proba, background)

            # 计算SHAP值
            shap_values = explainer.shap_values(X_processed)

            # 对于二分类，取正类的SHAP值
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            return {
                'shap_values': shap_values,
                'explainer': explainer,
                'feature_names': X.columns.tolist()
            }

        except Exception as e:
            logger.error(f"SHAP解释失败: {e}")
            return {}

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        交叉验证

        Args:
            X: 特征数据
            y: 标签数据
            cv_folds: 交叉验证折数

        Returns:
            Dict[str, Any]: 交叉验证结果
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_results = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"交叉验证第 {fold+1}/{cv_folds} 折")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # 训练模型
            fold_results = self.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            # 保存结果
            cv_results['fold_metrics'].append(fold_results['val_metrics'])

        # 计算均值和标准差
        metric_names = cv_results['fold_metrics'][0].keys()
        for metric in metric_names:
            values = [fold[metric] for fold in cv_results['fold_metrics']]
            cv_results['mean_metrics'][metric] = np.mean(values)
            cv_results['std_metrics'][metric] = np.std(values)

        logger.info(f"交叉验证完成: AUC={cv_results['mean_metrics']['auc']:.4f}±{cv_results['std_metrics']['auc']:.4f}")

        return cv_results

    def save_model(self, save_path: str):
        """保存模型"""
        import pickle

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'calibrator': self.calibrator,
            'config': self.config,
            'model_type': self.model_type
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存: {save_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        import pickle

        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.calibrator = model_data['calibrator']
        self.config = model_data['config']
        self.model_type = model_data['model_type']

        logger.info(f"模型已加载: {load_path}")

    def get_model_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        if self.model is None:
            return {}

        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return {'type': self.model_type}