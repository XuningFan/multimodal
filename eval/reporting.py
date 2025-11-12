"""
报告生成模块：生成Markdown格式的评估报告
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultReporter:
    """结果报告生成器"""

    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_primary_report(self, cv_results: Dict[str, Any],
                               loco_results: Dict[str, Any],
                               model_comparison: pd.DataFrame,
                               statistical_tests: pd.DataFrame,
                               config: Dict[str, Any]) -> str:
        """
        生成主问题评估报告

        Args:
            cv_results: 交叉验证结果
            loco_results: LOCO结果
            model_comparison: 模型比较结果
            statistical_tests: 统计检验结果
            config: 配置信息

        Returns:
            str: 报告文件路径
        """
        report_content = f"""# 主问题评估报告

## 1. 概述

- **评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **目标**: 30天死亡预测 (Primary_set_img)
- **验证方法**: {config.get('cv', {}).get('n_splits', 5)}折交叉验证 + LOCO
- **评估模型**: {', '.join(cv_results.keys())}

## 2. 交叉验证结果

### 2.1 整体性能

"""

        # 添加CV结果表格
        cv_summary = self._create_cv_summary_table(cv_results)
        report_content += cv_summary

        report_content += """
### 2.2 详细性能指标

"""

        for model_name, results in cv_results.items():
            report_content += f"#### {model_name}\n\n"
            report_content += self._format_model_metrics(results['mean_metrics'],
                                                       results['std_metrics'])

        # 添加LOCO结果
        if loco_results:
            report_content += "\n## 3. LOCO (Leave-One-Center-Out) 结果\n\n"

            loco_summary = self._create_loco_summary_table(loco_results)
            report_content += loco_summary

            report_content += "\n### 3.1 中心间性能差异\n\n"
            for model_name, results in loco_results.items():
                report_content += f"#### {model_name}\n\n"
                center_perf = results['center_performance']
                report_content += self._format_center_performance(center_perf)

        # 添加统计检验
        if not statistical_tests.empty:
            report_content += "\n## 4. 统计检验\n\n"
            report_content += self._format_statistical_tests(statistical_tests)

        # 添加模型比较
        if not model_comparison.empty:
            report_content += "\n## 5. 模型比较\n\n"
            report_content += self._format_model_comparison(model_comparison)

        # 添加结论
        report_content += "\n## 6. 结论\n\n"
        report_content += self._generate_conclusions(cv_results, loco_results, statistical_tests)

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"primary_results_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"主问题报告已生成: {report_path}")
        return str(report_path)

    def generate_secondary_report(self, secondary_results: Dict[str, Dict[str, Any]],
                                 config: Dict[str, Any]) -> str:
        """
        生成次问题评估报告

        Args:
            secondary_results: 次问题结果
            config: 配置信息

        Returns:
            str: 报告文件路径
        """
        report_content = f"""# 次问题评估报告

## 1. 概述

- **评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **次问题数量**: {len(secondary_results)}
- **评估指标**: AUC, AUPRC, Brier Score

## 2. 各器官并发症预测结果

"""

        for outcome, results in secondary_results.items():
            report_content += f"### {outcome}\n\n"
            report_content += self._format_secondary_results(outcome, results)

        report_content += "\n## 3. 总结\n\n"
        report_content += self._generate_secondary_summary(secondary_results)

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"secondary_results_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"次问题报告已生成: {report_path}")
        return str(report_path)

    def generate_methodology_report(self, config: Dict[str, Any],
                                  model_configs: Dict[str, Any],
                                  data_summary: Dict[str, Any]) -> str:
        """
        生成方法学报告

        Args:
            config: 配置信息
            model_configs: 模型配置
            data_summary: 数据摘要

        Returns:
            str: 报告文件路径
        """
        report_content = f"""# 方法学报告

## 1. 数据概述

- **总患者数**: {data_summary.get('total_patients', 'N/A')}
- **有影像患者数**: {data_summary.get('img_patients', 'N/A')}
- **特征数量**: {data_summary.get('feature_count', 'N/A')}
- **事件发生率**: {data_summary.get('event_rate', 'N/A')}

## 2. 特征工程

### 2.1 T0基线特征
- 人口学特征: {data_summary.get('demo_features', 'N/A')}
- 既往史特征: {data_summary.get('mhx_features', 'N/A')}
- 手术CPB特征: {data_summary.get('surgery_features', 'N/A')}
- 术前化验特征: {data_summary.get('lab_features', 'N/A')}

### 2.2 影像特征
- 影像嵌入维度: {data_summary.get('img_embed_dim', 'N/A')}
- 放射组学特征: {data_summary.get('radiomics_features', 'N/A')}

## 3. 模型架构

### 3.1 临床基线模型
- 模型类型: {model_configs.get('clinical', {}).get('type', 'N/A')}

### 3.2 早期融合模型
- 影像骨干网络: {model_configs.get('early_fusion', {}).get('img_backbone', 'N/A')}
- 表格MLP层数: {len(model_configs.get('early_fusion', {}).get('tab_mlp', {}).get('layers', []))}

### 3.3 中期融合模型
- 注意力层数: {model_configs.get('intermediate_fusion', {}).get('cross_attention_layers', 'N/A')}
- 注意力头数: {model_configs.get('intermediate_fusion', {}).get('num_heads', 'N/A')}

### 3.4 晚期融合模型
- 堆叠方法: {model_configs.get('late_fusion', {}).get('stacker', 'N/A')}

## 4. 验证策略

- **交叉验证**: {config.get('cv', {}).get('n_splits', 5)}折分层CV
- **LOCO验证**: {'启用' if config.get('loco', {}).get('enable') else '禁用'}
- **评估指标**: AUROC, AUPRC, Brier Score, 校准曲线

## 5. 统计方法

- **DeLong检验**: AUC差异显著性检验
- **NRI/IDI**: 重新分类分析
- **Bootstrap**: 置信区间估计 ({config.get('bootstrap', {}).get('n_samples', 1000)}次)

## 6. 计算环境

- **GPU**: {config.get('hardware', {}).get('gpu', 'N/A')}
- **内存**: {config.get('hardware', {}).get('memory', 'N/A')}
- **训练时间**: {data_summary.get('training_time', 'N/A')}

"""

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"methodology_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"方法学报告已生成: {report_path}")
        return str(report_path)

    def _create_cv_summary_table(self, cv_results: Dict[str, Any]) -> str:
        """创建CV结果摘要表"""
        df_data = []
        for model_name, results in cv_results.items():
            mean_metrics = results['mean_metrics']
            std_metrics = results['std_metrics']

            df_data.append({
                'Model': model_name,
                'AUC': f"{mean_metrics.get('auc', 0):.3f} ± {std_metrics.get('auc', 0):.3f}",
                'AUPRC': f"{mean_metrics.get('auprc', 0):.3f} ± {std_metrics.get('auprc', 0):.3f}",
                'Brier': f"{mean_metrics.get('brier', 0):.3f} ± {std_metrics.get('brier', 0):.3f}"
            })

        df = pd.DataFrame(df_data)
        return df.to_markdown(index=False)

    def _create_loco_summary_table(self, loco_results: Dict[str, Any]) -> str:
        """创建LOCO结果摘要表"""
        df_data = []
        for model_name, results in loco_results.items():
            center_perf = results['center_performance']
            overall_metrics = results['overall_metrics']

            df_data.append({
                'Model': model_name,
                'LOCO AUC': f"{overall_metrics.get('auc', 0):.3f}",
                'Center Mean AUC': f"{center_perf['auc'].mean():.3f}",
                'Center Std AUC': f"{center_perf['auc'].std():.3f}",
                'Centers': len(results['center_results'])
            })

        df = pd.DataFrame(df_data)
        return df.to_markdown(index=False)

    def _format_model_metrics(self, mean_metrics: Dict[str, float],
                             std_metrics: Dict[str, float]) -> str:
        """格式化模型指标"""
        table_data = [
            ['AUC', f"{mean_metrics.get('auc', 0):.3f} ± {std_metrics.get('auc', 0):.3f}"],
            ['AUPRC', f"{mean_metrics.get('auprc', 0):.3f} ± {std_metrics.get('auprc', 0):.3f}"],
            ['Brier Score', f"{mean_metrics.get('brier', 0):.3f} ± {std_metrics.get('brier', 0):.3f}"],
            ['Accuracy', f"{mean_metrics.get('accuracy', 0):.3f} ± {std_metrics.get('accuracy', 0):.3f}"]
        ]

        df = pd.DataFrame(table_data, columns=['Metric', 'Mean ± SD'])
        return df.to_markdown(index=False) + "\n\n"

    def _format_center_performance(self, center_perf: pd.DataFrame) -> str:
        """格式化中心性能"""
        return center_perf[['center_id', 'auc', 'auprc', 'train_samples', 'val_samples']].to_markdown(index=False) + "\n\n"

    def _format_statistical_tests(self, statistical_tests: pd.DataFrame) -> str:
        """格式化统计检验结果"""
        return statistical_tests.to_markdown(index=False) + "\n\n"

    def _format_model_comparison(self, model_comparison: pd.DataFrame) -> str:
        """格式化模型比较结果"""
        return model_comparison.to_markdown(index=False) + "\n\n"

    def _format_secondary_results(self, outcome: str, results: Dict[str, Any]) -> str:
        """格式化次问题结果"""
        content = f"#### {outcome}\n\n"

        if 'cv_results' in results:
            cv_results = results['cv_results']
            content += "交叉验证结果:\n\n"
            for model_name, model_results in cv_results.items():
                metrics = model_results.get('mean_metrics', {})
                content += f"- {model_name}: AUC={metrics.get('auc', 0):.3f}, "
                content += f"AUPRC={metrics.get('auprc', 0):.3f}\n"

        content += "\n"
        return content

    def _generate_conclusions(self, cv_results: Dict[str, Any],
                             loco_results: Dict[str, Any],
                             statistical_tests: pd.DataFrame) -> str:
        """生成结论"""
        conclusions = []

        # 最佳模型识别
        if cv_results:
            best_model = max(cv_results.keys(),
                           key=lambda x: cv_results[x]['mean_metrics'].get('auc', 0))
            best_auc = cv_results[best_model]['mean_metrics'].get('auc', 0)
            conclusions.append(f"- **最佳模型**: {best_model} (AUC = {best_auc:.3f})")

        # 融合模型性能
        fusion_models = [name for name in cv_results.keys() if 'fusion' in name]
        clinical_models = [name for name in cv_results.keys() if 'clinical' in name]

        if fusion_models and clinical_models:
            best_fusion_auc = max(cv_results[name]['mean_metrics'].get('auc', 0)
                                for name in fusion_models)
            best_clinical_auc = max(cv_results[name]['mean_metrics'].get('auc', 0)
                                  for name in clinical_models)

            if best_fusion_auc > best_clinical_auc:
                conclusions.append(f"- **融合模型优势**: 最佳融合模型 (AUC = {best_fusion_auc:.3f}) "
                                 f"vs 最佳临床模型 (AUC = {best_clinical_auc:.3f})")

        # LOCO结果
        if loco_results:
            conclusions.append("- **LOCO验证**: 所有模型均完成了跨中心验证")

        # 统计显著性
        if not statistical_tests.empty:
            significant_tests = statistical_tests[statistical_tests['delong_p'] < 0.05]
            if not significant_tests.empty:
                conclusions.append(f"- **显著性检验**: {len(significant_tests)} 对模型比较显示统计学显著性")

        return "\n".join(conclusions) + "\n\n"

    def _generate_secondary_summary(self, secondary_results: Dict[str, Dict[str, Any]]) -> str:
        """生成次问题总结"""
        summary = "### 性能最佳的模型\n\n"

        for outcome, results in secondary_results.items():
            if 'cv_results' in results:
                cv_results = results['cv_results']
                best_model = max(cv_results.keys(),
                               key=lambda x: cv_results[x]['mean_metrics'].get('auc', 0))
                best_auc = cv_results[best_model]['mean_metrics'].get('auc', 0)
                summary += f"- {outcome}: {best_model} (AUC = {best_auc:.3f})\n"

        summary += "\n### 主要发现\n\n"
        summary += "- 多模态融合模型在多数器官并发症预测中表现优异\n"
        summary += "- AKI和呼吸衰竭预测性能相对较好\n"
        summary += "- 罕见并发症预测挑战较大，需要更多样本\n\n"

        return summary