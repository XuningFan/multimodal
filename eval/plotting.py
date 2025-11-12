"""
结果可视化模块：绘制ROC曲线、校准曲线、决策曲线等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ResultPlotter:
    """结果可视化工具类"""

    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        初始化绘图器

        Args:
            style: 绘图风格
            figsize: 默认图形大小
        """
        self.style = style
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def plot_roc_curves(self, results_dict: Dict[str, Dict[str, Any]],
                       save_path: Optional[str] = None) -> None:
        """
        绘制多条ROC曲线

        Args:
            results_dict: 结果字典 {model_name: {'y_true': ..., 'y_pred_proba': ...}}
            save_path: 保存路径
        """
        plt.style.use(self.style)
        plt.figure(figsize=self.figsize)

        for i, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']

            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})',
                    color=self.colors[i % len(self.colors)])

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_calibration_curves(self, results_dict: Dict[str, Dict[str, Any]],
                               save_path: Optional[str] = None) -> None:
        """
        绘制校准曲线

        Args:
            results_dict: 结果字典
            save_path: 保存路径
        """
        from sklearn.calibration import calibration_curve

        plt.style.use(self.style)
        plt.figure(figsize=self.figsize)

        for i, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']

            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )

            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f'{model_name}', linewidth=2, markersize=6,
                    color=self.colors[i % len(self.colors)])

        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves', fontsize=14)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_decision_curves(self, decision_data: Dict[str, pd.DataFrame],
                           save_path: Optional[str] = None) -> None:
        """
        绘制决策曲线

        Args:
            decision_data: 决策分析数据
            save_path: 保存路径
        """
        plt.style.use(self.style)
        plt.figure(figsize=self.figsize)

        for i, (model_name, df) in enumerate(decision_data.items()):
            plt.plot(df['threshold'], df['net_benefit'], linewidth=2,
                    label=model_name, color=self.colors[i % len(self.colors)])

        # 参考线
        if decision_data:
            thresholds = list(decision_data.values())[0]['threshold']
            plt.plot(thresholds, thresholds - 1, 'k--', linewidth=1, label='Treat All')
            plt.plot(thresholds, np.zeros_like(thresholds), 'k-', linewidth=1, label='Treat None')

        plt.xlim([0.0, 1.0])
        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title('Decision Curve Analysis', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             metric: str = 'auc',
                             save_path: Optional[str] = None) -> None:
        """
        绘制模型比较图

        Args:
            comparison_df: 比较数据框
            metric: 比较指标
            save_path: 保存路径
        """
        plt.style.use(self.style)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 条形图
        sns.barplot(data=comparison_df, x='model', y=metric, ax=axes[0])
        axes[0].set_title(f'Model Comparison - {metric.upper()}', fontsize=14)
        axes[0].set_ylabel(metric.upper(), fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)

        # 误差条图
        if 'std' in comparison_df.columns:
            axes[1].errorbar(comparison_df['model'], comparison_df[metric],
                           yerr=comparison_df['std'], fmt='o', capsize=5, capthick=2)
            axes[1].set_title(f'Model Comparison with Error Bars - {metric.upper()}', fontsize=14)
            axes[1].set_ylabel(metric.upper(), fontsize=12)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_loco_forest_plot(self, loco_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> None:
        """
        绘制LOCO森林图

        Args:
            loco_results: LOCO结果
            save_path: 保存路径
        """
        if not loco_results:
            logger.warning("没有LOCO结果可绘制")
            return

        n_models = len(loco_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))

        if n_models == 1:
            axes = [axes]

        for i, (model_name, results) in enumerate(loco_results.items()):
            center_perf = results['center_performance']
            overall_auc = results['overall_metrics']['auc']

            centers = center_perf['center_id'].tolist()
            aucs = center_perf['auc'].tolist()
            y_pos = np.arange(len(centers))

            # 简化的置信区间
            ci_lower = np.maximum(aucs - 0.05, 0)
            ci_upper = np.minimum(aucs + 0.05, 1)

            axes[i].errorbar(aucs, y_pos, xerr=[aucs - ci_lower, ci_upper - aucs],
                           fmt='o', capsize=5, capthick=2, markersize=8)
            axes[i].axvline(overall_auc, color='red', linestyle='--',
                           label=f'Overall AUC: {overall_auc:.3f}')
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(centers)
            axes[i].set_xlabel('AUC', fontsize=12)
            axes[i].set_title(f'LOCO Performance - {model_name}', fontsize=14)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_feature_importance(self, feature_importance: Dict[str, float],
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """
        绘制特征重要性图

        Args:
            feature_importance: 特征重要性字典
            top_n: 显示前N个重要特征
            save_path: 保存路径
        """
        # 排序并选择前N个特征
        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:top_n]

        features, importance = zip(*sorted_features)

        plt.style.use(self.style)
        plt.figure(figsize=(10, 8))

        # 创建水平条形图
        y_pos = np.arange(len(features))
        colors = ['red' if imp < 0 else 'blue' for imp in importance]

        plt.barh(y_pos, importance, color=colors, alpha=0.7)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')

        # 添加数值标签
        for i, (feat, imp) in enumerate(zip(features, importance)):
            plt.text(imp + 0.001 * max(importance), i, f'{imp:.3f}',
                    va='center', ha='left' if imp > 0 else 'right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_learning_curves(self, train_losses: List[float],
                           val_losses: List[float],
                           train_metrics: Optional[List[float]] = None,
                           val_metrics: Optional[List[float]] = None,
                           metric_name: str = 'AUC',
                           save_path: Optional[str] = None) -> None:
        """
        绘制学习曲线

        Args:
            train_losses: 训练损失
            val_losses: 验证损失
            train_metrics: 训练指标
            val_metrics: 验证指标
            metric_name: 指标名称
            save_path: 保存路径
        """
        plt.style.use(self.style)

        if train_metrics is not None and val_metrics is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # 损失曲线
            axes[0].plot(train_losses, label='Train Loss', linewidth=2)
            axes[0].plot(val_losses, label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training and Validation Loss', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 指标曲线
            axes[1].plot(train_metrics, label=f'Train {metric_name}', linewidth=2)
            axes[1].plot(val_metrics, label=f'Val {metric_name}', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel(metric_name, fontsize=12)
            axes[1].set_title(f'Training and Validation {metric_name}', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        else:
            plt.figure(figsize=self.figsize)
            plt.plot(train_losses, label='Train Loss', linewidth=2)
            plt.plot(val_losses, label='Val Loss', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training and Validation Loss', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()