"""
模型工具函数
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelUtils:
    """模型工具类"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                         y_pred_class: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            y_pred_class: 预测类别

        Returns:
            Dict[str, float]: 指标字典
        """
        if y_pred_class is None:
            y_pred_class = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'auprc': average_precision_score(y_true, y_pred_proba),
            'brier': brier_score_loss(y_true, y_pred_proba),
            'accuracy': np.mean(y_true == y_pred_class)
        }

        # 计算敏感性和特异性
        tp = np.sum((y_true == 1) & (y_pred_class == 1))
        tn = np.sum((y_true == 0) & (y_pred_class == 0))
        fp = np.sum((y_true == 0) & (y_pred_class == 1))
        fn = np.sum((y_true == 1) & (y_pred_class == 0))

        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['recall'] = metrics['sensitivity']

        # F1分数
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (
                metrics['precision'] + metrics['recall']
            )
        else:
            metrics['f1'] = 0.0

        return metrics

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """
        绘制ROC曲线

        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            model_name: 模型名称
            save_path: 保存路径
        """
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """
        绘制精确率-召回率曲线

        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            model_name: 模型名称
            save_path: 保存路径
        """
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AUPRC = {auprc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             model_name: str = "Model", n_bins: int = 10,
                             save_path: Optional[str] = None) -> None:
        """
        绘制校准曲线

        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            model_name: 模型名称
            n_bins: 分箱数量
            save_path: 保存路径
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )

        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{model_name}')
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def create_confusion_matrix(y_true: np.ndarray, y_pred_class: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> pd.DataFrame:
        """
        创建混淆矩阵

        Args:
            y_true: 真实标签
            y_pred_class: 预测类别
            class_names: 类别名称
            save_path: 保存路径

        Returns:
            pd.DataFrame: 混淆矩阵
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred_class)

        if class_names is None:
            class_names = ['Negative', 'Positive']

        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        # 绘制热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        return cm_df

    @staticmethod
    def bootstrap_confidence_interval(metric_func, y_true: np.ndarray,
                                    y_pred_proba: np.ndarray,
                                    n_bootstrap: int = 1000,
                                    ci_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap置信区间计算

        Args:
            metric_func: 指标计算函数
            y_true: 真实标签
            y_pred_proba: 预测概率
            n_bootstrap: Bootstrap次数
            ci_level: 置信水平

        Returns:
            Tuple[float, float, float]: (均值, 下限, 上限)
        """
        np.random.seed(42)
        n_samples = len(y_true)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Bootstrap采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            if len(np.unique(y_true[indices])) < 2:  # 确保有正负样本
                continue

            score = metric_func(y_true[indices], y_pred_proba[indices])
            bootstrap_scores.append(score)

        bootstrap_scores = np.array(bootstrap_scores)

        # 计算置信区间
        alpha = 1 - ci_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        mean_score = np.mean(bootstrap_scores)

        return mean_score, ci_lower, ci_upper

    @staticmethod
    def calculate_nri(y_true: np.ndarray, prob_old: np.ndarray,
                     prob_new: np.ndarray) -> Tuple[float, float, float]:
        """
        计算净重新分类指数 (NRI)

        Args:
            y_true: 真实标签
            prob_old: 旧模型预测概率
            prob_new: 新模型预测概率

        Returns:
            Tuple[float, float, float]: (NRI, 事件NRI, 非事件NRI)
        """
        # 事件组（y=1）
        event_mask = y_true == 1
        n_events = event_mask.sum()

        if n_events > 0:
            event_risk_new = prob_new[event_mask]
            event_risk_old = prob_old[event_mask]
            events_upgraded = (event_risk_new > event_risk_old).sum()
            events_downgraded = (event_risk_new < event_risk_old).sum()
            nri_events = (events_upgraded - events_downgraded) / n_events
        else:
            nri_events = 0.0

        # 非事件组（y=0）
        nonevent_mask = y_true == 0
        n_nonevents = nonevent_mask.sum()

        if n_nonevents > 0:
            nonevent_risk_new = prob_new[nonevent_mask]
            nonevent_risk_old = prob_old[nonevent_mask]
            nonevents_upgraded = (nonevent_risk_new > nonevent_risk_old).sum()
            nonevents_downgraded = (nonevent_risk_new < nonevent_risk_old).sum()
            nri_nonevents = (nonevents_downgraded - nonevents_upgraded) / n_nonevents
        else:
            nri_nonevents = 0.0

        # 总体NRI
        nri_total = nri_events + nri_nonevents

        return nri_total, nri_events, nri_nonevents

    @staticmethod
    def calculate_decision_curve_analytics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                         threshold_range: np.ndarray = None) -> pd.DataFrame:
        """
        计算决策曲线分析

        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            threshold_range: 阈值范围

        Returns:
            pd.DataFrame: 决策曲线分析结果
        """
        if threshold_range is None:
            threshold_range = np.arange(0.01, 0.99, 0.01)

        n = len(y_true)
        n_events = y_true.sum()
        n_nonevents = n - n_events

        results = []

        for threshold in threshold_range:
            # 计算真阳性、假阳性等
            tp = ((y_pred_proba >= threshold) & (y_true == 1)).sum()
            fp = ((y_pred_proba >= threshold) & (y_true == 0)).sum()
            fn = ((y_pred_proba < threshold) & (y_true == 1)).sum()
            tn = ((y_pred_proba < threshold) & (y_true == 0)).sum()

            # 净收益
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))

            # 干预避免率（与全部干预相比）
            net_intervention_avoidance = 1 - (tp + fp) / n_events

            results.append({
                'threshold': threshold,
                'net_benefit': net_benefit,
                'net_intervention_avoidance': net_intervention_avoidance,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })

        return pd.DataFrame(results)

    @staticmethod
    def plot_decision_curve(df_decision: pd.DataFrame, model_name: str = "Model",
                           save_path: Optional[str] = None) -> None:
        """
        绘制决策曲线

        Args:
            df_decision: 决策曲线分析数据框
            model_name: 模型名称
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))

        # 模型决策曲线
        plt.plot(df_decision['threshold'], df_decision['net_benefit'],
                label=model_name, linewidth=2)

        # 参考线
        plt.plot(df_decision['threshold'], df_decision['threshold'] - 1,
                'k--', label='Treat All', linewidth=1)
        plt.plot(df_decision['threshold'], np.zeros_like(df_decision['threshold']),
                'k-', label='Treat None', linewidth=1)

        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        plt.title(f'Decision Curve Analysis - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()