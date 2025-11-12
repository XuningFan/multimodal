"""
交叉验证和LOCO（Leave-One-Center-Out）评估模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class CVLOCOEvaluator:
    """交叉验证和LOCO评估器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器

        Args:
            config: 评估配置
        """
        self.config = config
        self.cv_folds = config.get('cv_folds', 5)
        self.random_state = config.get('random_state', 42)
        self.loco_enabled = config.get('loco', {}).get('enable', False)
        self.center_field = config.get('loco', {}).get('center_field', 'center_id')

        # 存储结果
        self.cv_results = {}
        self.loco_results = {}
        self.fold_predictions = []
        self.loco_predictions = []

    def evaluate_cv(self, X: pd.DataFrame, y: pd.Series,
                   models: Dict[str, Any],
                   feature_groups: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        执行交叉验证评估

        Args:
            X: 特征数据
            y: 标签数据
            models: 模型字典
            feature_groups: 特征分组

        Returns:
            Dict[str, Any]: CV评估结果
        """
        logger.info(f"开始{self.cv_folds}折交叉验证评估")

        # 设置交叉验证
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        all_results = {}

        for model_name, model in models.items():
            logger.info(f"评估模型: {model_name}")

            model_results = {
                'fold_metrics': [],
                'fold_predictions': [],
                'feature_importance': [],
                'mean_metrics': {},
                'std_metrics': {},
                'confusion_matrices': []
            }

            # 执行交叉验证
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                fold_result = self._evaluate_single_fold(
                    X, y, model, train_idx, val_idx, fold, model_name
                )
                model_results['fold_metrics'].append(fold_result['metrics'])
                model_results['fold_predictions'].append(fold_result['predictions'])
                if 'feature_importance' in fold_result:
                    model_results['feature_importance'].append(fold_result['feature_importance'])
                if 'confusion_matrix' in fold_result:
                    model_results['confusion_matrices'].append(fold_result['confusion_matrix'])

            # 计算总体指标
            model_results['mean_metrics'], model_results['std_metrics'] = self._compute_cv_summary(
                model_results['fold_metrics']
            )

            # 合并所有fold的预测
            all_predictions = self._combine_fold_predictions(model_results['fold_predictions'])
            model_results['all_predictions'] = all_predictions

            # 计算整体指标
            overall_metrics = self._calculate_overall_metrics(y, all_predictions)
            model_results['overall_metrics'] = overall_metrics

            all_results[model_name] = model_results

        self.cv_results = all_results
        logger.info("交叉验证评估完成")

        return all_results

    def evaluate_loco(self, X: pd.DataFrame, y: pd.Series, center_ids: pd.Series,
                     models: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行LOCO评估

        Args:
            X: 特征数据
            y: 标签数据
            center_ids: 中心ID
            models: 模型字典

        Returns:
            Dict[str, Any]: LOCO评估结果
        """
        if not self.loco_enabled:
            logger.info("LOCO评估未启用")
            return {}

        logger.info("开始LOCO评估")

        # 获取唯一的中心
        unique_centers = center_ids.unique()
        logger.info(f"发现 {len(unique_centers)} 个中心")

        all_results = {}

        for model_name, model in models.items():
            logger.info(f"LOCO评估模型: {model_name}")

            model_results = {
                'center_results': {},
                'center_predictions': {},
                'overall_metrics': {},
                'center_performance': pd.DataFrame()
            }

            # 对每个中心进行留一验证
            for center in unique_centers:
                logger.info(f"留出中心: {center}")

                # 分割数据
                train_mask = center_ids != center
                val_mask = center_ids == center

                X_train, X_val = X[train_mask], X[val_mask]
                y_train, y_val = y[train_mask], y[val_mask]

                # 训练和评估
                center_result = self._evaluate_loco_center(
                    X_train, y_train, X_val, y_val, model, center
                )

                model_results['center_results'][center] = center_result['metrics']
                model_results['center_predictions'][center] = center_result['predictions']

                # 记录中心性能
                center_perf = {
                    'center_id': center,
                    'train_samples': len(y_train),
                    'val_samples': len(y_val),
                    **center_result['metrics']
                }
                model_results['center_performance'] = pd.concat([
                    model_results['center_performance'],
                    pd.DataFrame([center_perf])
                ], ignore_index=True)

            # 计算LOCO总体指标
            all_predictions = self._combine_loco_predictions(model_results['center_predictions'])
            all_labels = []
            for center in unique_centers:
                val_mask = center_ids == center
                all_labels.extend(y[val_mask])

            overall_metrics = self._calculate_overall_metrics(np.array(all_labels), all_predictions)
            model_results['overall_metrics'] = overall_metrics

            all_results[model_name] = model_results

        self.loco_results = all_results
        logger.info("LOCO评估完成")

        return all_results

    def _evaluate_single_fold(self, X: pd.DataFrame, y: pd.Series,
                             model: Any, train_idx: np.ndarray, val_idx: np.ndarray,
                             fold: int, model_name: str) -> Dict[str, Any]:
        """评估单个fold"""
        # 分割数据
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 克隆模型（确保独立性）
        import copy
        fold_model = copy.deepcopy(model)

        # 训练模型
        if hasattr(fold_model, 'train'):
            # 自定义模型训练方法
            training_result = fold_model.train(X_train, y_train)
        else:
            # sklearn风格模型
            fold_model.fit(X_train, y_train)

        # 预测
        if hasattr(fold_model, 'predict'):
            predictions = fold_model.predict(X_val)
            if hasattr(fold_model, 'predict_proba'):
                probas = fold_model.predict_proba(X_val)[:, 1]
            else:
                probas = predictions
        else:
            # 自定义预测方法
            probas, predictions = fold_model.predict(X_val)

        # 计算指标
        metrics = self._calculate_fold_metrics(y_val, probas, predictions)

        result = {
            'fold': fold,
            'metrics': metrics,
            'predictions': {
                'y_true': y_val.values,
                'y_pred_proba': probas,
                'y_pred_class': predictions
            }
        }

        # 特征重要性
        if hasattr(fold_model, 'get_feature_importance'):
            result['feature_importance'] = fold_model.get_feature_importance()

        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        result['confusion_matrix'] = confusion_matrix(y_val, predictions)

        return result

    def _evaluate_loco_center(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             model: Any, center: str) -> Dict[str, Any]:
        """评估单个中心的LOCO"""
        # 训练模型
        import copy
        loco_model = copy.deepcopy(model)

        if hasattr(loco_model, 'train'):
            training_result = loco_model.train(X_train, y_train)
        else:
            loco_model.fit(X_train, y_train)

        # 预测
        if hasattr(loco_model, 'predict'):
            predictions = loco_model.predict(X_val)
            if hasattr(loco_model, 'predict_proba'):
                probas = loco_model.predict_proba(X_val)[:, 1]
            else:
                probas = predictions
        else:
            probas, predictions = loco_model.predict(X_val)

        # 计算指标
        metrics = self._calculate_fold_metrics(y_val, probas, predictions)

        return {
            'center': center,
            'metrics': metrics,
            'predictions': {
                'y_true': y_val.values,
                'y_pred_proba': probas,
                'y_pred_class': predictions
            }
        }

    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               y_pred_class: np.ndarray) -> Dict[str, float]:
        """计算fold指标"""
        return {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'auprc': average_precision_score(y_true, y_pred_proba),
            'brier': brier_score_loss(y_true, y_pred_proba),
            'accuracy': np.mean(y_true == y_pred_class)
        }

    def _compute_cv_summary(self, fold_metrics: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """计算交叉验证总结"""
        if not fold_metrics:
            return {}, {}

        metric_names = fold_metrics[0].keys()
        mean_metrics = {}
        std_metrics = {}

        for metric in metric_names:
            values = [fold[metric] for fold in fold_metrics]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)

        return mean_metrics, std_metrics

    def _combine_fold_predictions(self, fold_predictions: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """合并所有fold的预测"""
        combined = {
            'y_true': np.array([]),
            'y_pred_proba': np.array([]),
            'y_pred_class': np.array([])
        }

        for fold_pred in fold_predictions:
            combined['y_true'] = np.concatenate([combined['y_true'], fold_pred['y_true']])
            combined['y_pred_proba'] = np.concatenate([combined['y_pred_proba'], fold_pred['y_pred_proba']])
            combined['y_pred_class'] = np.concatenate([combined['y_pred_class'], fold_pred['y_pred_class']])

        return combined

    def _combine_loco_predictions(self, center_predictions: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """合并所有中心的预测"""
        all_probas = np.array([])

        for center, predictions in center_predictions.items():
            all_probas = np.concatenate([all_probas, predictions['y_pred_proba']])

        return all_probas

    def _calculate_overall_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """计算整体指标"""
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        return self._calculate_fold_metrics(y_true, y_pred_proba, y_pred_class)

    def plot_loco_forest(self, save_path: Optional[str] = None) -> None:
        """绘制LOCO森林图"""
        if not self.loco_results:
            logger.warning("没有LOCO结果可绘制")
            return

        fig, axes = plt.subplots(1, len(self.loco_results), figsize=(15, 6))
        if len(self.loco_results) == 1:
            axes = [axes]

        for i, (model_name, results) in enumerate(self.loco_results.items()):
            center_perf = results['center_performance']
            overall_auc = results['overall_metrics']['auc']

            # 创建森林图数据
            centers = center_perf['center_id'].tolist()
            aucs = center_perf['auc'].tolist()
            ci_lower = aucs - 0.05  # 简化的置信区间
            ci_upper = aucs + 0.05

            y_pos = np.arange(len(centers))

            # 绘制森林图
            axes[i].errorbar(aucs, y_pos, xerr=[aucs - ci_lower, ci_upper - aucs],
                           fmt='o', capsize=5, capthick=2)
            axes[i].axvline(overall_auc, color='red', linestyle='--', label=f'Overall AUC: {overall_auc:.3f}')
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(centers)
            axes[i].set_xlabel('AUC')
            axes[i].set_title(f'LOCO Performance - {model_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def compare_models(self, metric: str = 'auc') -> pd.DataFrame:
        """
        比较模型性能

        Args:
            metric: 比较的指标

        Returns:
            pd.DataFrame: 模型比较结果
        """
        comparison_data = []

        # CV结果比较
        if self.cv_results:
            for model_name, results in self.cv_results.items():
                mean_val = results['mean_metrics'].get(metric, 0)
                std_val = results['std_metrics'].get(metric, 0)
                overall_val = results['overall_metrics'].get(metric, 0)

                comparison_data.append({
                    'model': model_name,
                    'method': 'CV',
                    'mean': mean_val,
                    'std': std_val,
                    'overall': overall_val
                })

        # LOCO结果比较
        if self.loco_results:
            for model_name, results in self.loco_results.items():
                overall_val = results['overall_metrics'].get(metric, 0)

                # 计算中心间性能差异
                center_perfs = results['center_performance'][metric]
                center_mean = center_perfs.mean()
                center_std = center_perfs.std()

                comparison_data.append({
                    'model': model_name,
                    'method': 'LOCO',
                    'mean': center_mean,
                    'std': center_std,
                    'overall': overall_val
                })

        return pd.DataFrame(comparison_data)

    def generate_summary_report(self) -> Dict[str, Any]:
        """生成评估总结报告"""
        summary = {
            'evaluation_config': self.config,
            'cv_summary': {},
            'loco_summary': {},
            'model_comparison': {}
        }

        # CV总结
        if self.cv_results:
            summary['cv_summary'] = {
                'models_evaluated': list(self.cv_results.keys()),
                'best_cv_model': None,
                'best_cv_metric': 0
            }

            for model_name, results in self.cv_results.items():
                auc = results['mean_metrics'].get('auc', 0)
                if auc > summary['cv_summary']['best_cv_metric']:
                    summary['cv_summary']['best_cv_model'] = model_name
                    summary['cv_summary']['best_cv_metric'] = auc

        # LOCO总结
        if self.loco_results:
            summary['loco_summary'] = {
                'models_evaluated': list(self.loco_results.keys()),
                'best_loco_model': None,
                'best_loco_metric': 0,
                'centers': len(self.loco_results[list(self.loco_results.keys())[0]]['center_results'])
            }

            for model_name, results in self.loco_results.items():
                auc = results['overall_metrics'].get('auc', 0)
                if auc > summary['loco_summary']['best_loco_metric']:
                    summary['loco_summary']['best_loco_model'] = model_name
                    summary['loco_summary']['best_loco_metric'] = auc

        # 模型比较
        summary['model_comparison'] = self.compare_models()

        return summary