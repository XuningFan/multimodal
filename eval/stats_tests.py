"""
统计检验模块：DeLong检验、NRI/IDI计算、显著性检验
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

logger = logging.getLogger(__name__)


class StatisticalTests:
    """统计检验工具类"""

    @staticmethod
    def delong_test(y_true: np.ndarray, y_pred_proba1: np.ndarray,
                   y_pred_proba2: np.ndarray) -> Tuple[float, float, float]:
        """
        DeLong检验：比较两个AUC的差异

        Args:
            y_true: 真实标签
            y_pred_proba1: 模型1预测概率
            y_pred_proba2: 模型2预测概率

        Returns:
            Tuple[float, float, float]: (z_statistic, p_value, auc_difference)
        """
        try:
            # 计算AUC
            auc1 = roc_auc_score(y_true, y_pred_proba1)
            auc2 = roc_auc_score(y_true, y_pred_proba2)

            # 计算DeLong统计量
            z_stat, p_value = StatisticalTests._delong_roc_test(
                y_true, y_pred_proba1, y_pred_proba2
            )

            auc_diff = auc1 - auc2

            return z_stat, p_value, auc_diff

        except Exception as e:
            logger.error(f"DeLong检验失败: {e}")
            return 0.0, 1.0, 0.0

    @staticmethod
    def _delong_roc_test(y_true: np.ndarray, y_pred_proba1: np.ndarray,
                        y_pred_proba2: np.ndarray) -> Tuple[float, float]:
        """
        DeLong ROC检验的实现
        """
        # 获取正负样本索引
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]

        n_pos = len(pos_idx)
        n_neg = len(neg_idx)

        # 计算结构化协方差矩阵
        def _auc_components(y_true, y_pred_proba):
            """计算AUC的组成部分"""
            pos_scores = y_pred_proba[pos_idx]
            neg_scores = y_pred_proba[neg_idx]

            # 计算结构化AUC
            V10 = np.mean(np.array([np.mean(pos_scores > neg_score) for neg_score in neg_scores]))
            V01 = np.mean(np.array([np.mean(neg_scores < pos_score) for pos_score in pos_scores]))

            # 计算方差成分
            S10 = np.array([np.mean(pos_scores > neg_score) for neg_score in neg_scores])
            S01 = np.array([np.mean(neg_scores < pos_score) for pos_score in pos_scores])

            return V10, V01, S10, S01

        # 计算两个模型的组成部分
        V10_1, V01_1, S10_1, S01_1 = _auc_components(y_true, y_pred_proba1)
        V10_2, V01_2, S10_2, S01_2 = _auc_components(y_true, y_pred_proba2)

        # 计算协方差
        S10 = S10_1 - S10_2
        S01 = S01_1 - S01_2

        # 计算方差
        var_S10 = np.var(S10) / n_neg
        var_S01 = np.var(S01) / n_pos

        # 计算总方差
        var_diff = var_S10 + var_S01

        # 计算z统计量和p值
        if var_diff > 0:
            auc1 = roc_auc_score(y_true, y_pred_proba1)
            auc2 = roc_auc_score(y_true, y_pred_proba2)
            z_stat = (auc1 - auc2) / np.sqrt(var_diff)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0.0
            p_value = 1.0

        return z_stat, p_value

    @staticmethod
    def calculate_nri(y_true: np.ndarray, prob_old: np.ndarray,
                     prob_new: np.ndarray, category_cutoffs: Optional[List[float]] = None) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        计算净重新分类指数 (Net Reclassification Index, NRI)

        Args:
            y_true: 真实标签
            prob_old: 旧模型预测概率
            prob_new: 新模型预测概率
            category_cutoffs: 分类阈值列表

        Returns:
            Tuple[float, float, float, Dict]: (总NRI, 事件NRI, 非事件NRI, 详细统计)
        """
        if category_cutoffs is None:
            category_cutoffs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # 将概率转换为分类
        cat_old = np.digitize(prob_old, category_cutoffs) - 1
        cat_new = np.digitize(prob_new, category_cutoffs) - 1

        # 分离事件和非事件
        event_mask = y_true == 1
        nonevent_mask = ~event_mask

        # 事件组重新分类
        event_cat_old = cat_old[event_mask]
        event_cat_new = cat_new[event_mask]

        # 计算事件组的移动
        event_up = np.sum(event_cat_new > event_cat_old)
        event_down = np.sum(event_cat_new < event_cat_old)
        event_same = np.sum(event_cat_new == event_cat_old)

        n_events = len(event_cat_old)
        nri_events = (event_up - event_down) / n_events if n_events > 0 else 0.0

        # 非事件组重新分类
        nonevent_cat_old = cat_old[nonevent_mask]
        nonevent_cat_new = cat_new[nonevent_mask]

        # 计算非事件组的移动（向下移动为改善）
        nonevent_down = np.sum(nonevent_cat_new > nonevent_cat_old)
        nonevent_up = np.sum(nonevent_cat_new < nonevent_cat_old)
        nonevent_same = np.sum(nonevent_cat_new == nonevent_cat_old)

        n_nonevents = len(nonevent_cat_old)
        nri_nonevents = (nonevent_down - nonevent_up) / n_nonevents if n_nonevents > 0 else 0.0

        # 总体NRI
        nri_total = nri_events + nri_nonevents

        # 详细统计
        details = {
            'events': {
                'n': n_events,
                'upgraded': int(event_up),
                'downgraded': int(event_down),
                'same': int(event_same),
                'nri': nri_events
            },
            'nonevents': {
                'n': n_nonevents,
                'upgraded': int(nonevent_up),
                'downgraded': int(nonevent_down),
                'same': int(nonevent_same),
                'nri': nri_nonevents
            },
            'categories': category_cutoffs
        }

        return nri_total, nri_events, nri_nonevents, details

    @staticmethod
    def calculate_idi(y_true: np.ndarray, prob_old: np.ndarray,
                     prob_new: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
        """
        计算净判别改善指数 (Integrated Discrimination Improvement, IDI)

        Args:
            y_true: 真实标签
            prob_old: 旧模型预测概率
            prob_new: 新模型预测概率

        Returns:
            Tuple[float, float, Dict]: (IDI, 事件IDI, 详细统计)
        """
        # 分离事件和非事件
        event_mask = y_true == 1
        nonevent_mask = ~event_mask

        # 计算平均预测概率的差异
        prob_old_events = prob_old[event_mask]
        prob_new_events = prob_new[event_mask]
        prob_old_nonevents = prob_old[nonevent_mask]
        prob_new_nonevents = prob_new[nonevent_mask]

        # 事件组的平均预测概率变化
        mean_old_events = np.mean(prob_old_events)
        mean_new_events = np.mean(prob_new_events)
        diff_events = mean_new_events - mean_old_events

        # 非事件组的平均预测概率变化
        mean_old_nonevents = np.mean(prob_old_nonevents)
        mean_new_nonevents = np.mean(prob_new_nonevents)
        diff_nonevents = mean_old_nonevents - mean_new_nonevents  # 注意方向

        # IDI
        idi = diff_events + diff_nonevents

        # 详细统计
        details = {
            'events': {
                'n': len(prob_old_events),
                'mean_old': mean_old_events,
                'mean_new': mean_new_events,
                'diff': diff_events
            },
            'nonevents': {
                'n': len(prob_old_nonevents),
                'mean_old': mean_old_nonevents,
                'mean_new': mean_new_nonevents,
                'diff': diff_nonevents
            },
            'idi_components': {
                'events_component': diff_events,
                'nonevents_component': diff_nonevents,
                'total_idi': idi
            }
        }

        return idi, diff_events, details

    @staticmethod
    def bootstrap_test_statistic(y_true: np.ndarray, prob_old: np.ndarray,
                                prob_new: np.ndarray, test_type: str = 'delong',
                                n_bootstrap: int = 1000,
                                random_state: int = 42) -> Dict[str, Any]:
        """
        Bootstrap统计检验

        Args:
            y_true: 真实标签
            prob_old: 旧模型预测概率
            prob_new: 新模型预测概率
            test_type: 检验类型 ('delong', 'nri', 'idi')
            n_bootstrap: Bootstrap次数
            random_state: 随机种子

        Returns:
            Dict[str, Any]: Bootstrap检验结果
        """
        np.random.seed(random_state)
        n_samples = len(y_true)

        bootstrap_stats = []

        # 计算观测统计量
        if test_type == 'delong':
            observed_stat, observed_p, observed_diff = StatisticalTests.delong_test(
                y_true, prob_old, prob_new
            )
        elif test_type == 'nri':
            observed_stat, _, _, _ = StatisticalTests.calculate_nri(
                y_true, prob_old, prob_new
            )
        elif test_type == 'idi':
            observed_stat, _, _ = StatisticalTests.calculate_idi(
                y_true, prob_old, prob_new
            )
        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        # Bootstrap采样
        for i in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)

            if len(np.unique(y_true[indices])) < 2:  # 确保有正负样本
                continue

            y_boot = y_true[indices]
            prob_old_boot = prob_old[indices]
            prob_new_boot = prob_new[indices]

            try:
                if test_type == 'delong':
                    boot_stat, _, _ = StatisticalTests.delong_test(
                        y_boot, prob_old_boot, prob_new_boot
                    )
                elif test_type == 'nri':
                    boot_stat, _, _, _ = StatisticalTests.calculate_nri(
                        y_boot, prob_old_boot, prob_new_boot
                    )
                elif test_type == 'idi':
                    boot_stat, _, _ = StatisticalTests.calculate_idi(
                        y_boot, prob_old_boot, prob_new_boot
                    )

                bootstrap_stats.append(boot_stat)

            except Exception as e:
                warnings.warn(f"Bootstrap iteration {i} failed: {e}")
                continue

        # 计算置信区间
        bootstrap_stats = np.array(bootstrap_stats)
        if len(bootstrap_stats) > 0:
            ci_lower = np.percentile(bootstrap_stats, 2.5)
            ci_upper = np.percentile(bootstrap_stats, 97.5)
            p_value_bootstrap = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
        else:
            ci_lower, ci_upper = observed_stat, observed_stat
            p_value_bootstrap = 1.0

        return {
            'test_type': test_type,
            'observed_statistic': observed_stat,
            'bootstrap_mean': np.mean(bootstrap_stats) if len(bootstrap_stats) > 0 else observed_stat,
            'bootstrap_std': np.std(bootstrap_stats) if len(bootstrap_stats) > 0 else 0.0,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value_bootstrap': p_value_bootstrap,
            'n_successful_bootstrap': len(bootstrap_stats),
            'n_total_bootstrap': n_bootstrap
        }

    @staticmethod
    def multiple_comparison_correction(p_values: List[float], method: str = 'bonferroni') -> List[float]:
        """
        多重比较校正

        Args:
            p_values: p值列表
            method: 校正方法 ('bonferroni', 'fdr_bh', 'fdr_by')

        Returns:
            List[float]: 校正后的p值
        """
        p_values = np.array(p_values)

        if method == 'bonferroni':
            corrected_p = p_values * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)  # 确保不超过1

        elif method in ['fdr_bh', 'fdr_by']:
            from statsmodels.stats.multitest import multipletests
            corrected_p = multipletests(p_values, method=method[4:])[1]

        else:
            raise ValueError(f"Unsupported correction method: {method}")

        return corrected_p.tolist()

    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05, power: float = 0.8,
                      sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        统计功效分析

        Args:
            effect_size: 效应量
            alpha: 显著性水平
            power: 统计功效
            sample_size: 样本量（可选）

        Returns:
            Dict[str, Any]: 功效分析结果
        """
        try:
            from statsmodels.stats.power import NormalIndPower
        except ImportError:
            logger.warning("statsmodels未安装，无法进行功效分析")
            return {}

        power_analysis = NormalIndPower()

        if sample_size is None:
            # 计算所需样本量
            n_required = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
            return {
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'required_sample_size': n_required
            }
        else:
            # 计算统计功效
            calculated_power = power_analysis.power(
                effect_size=effect_size,
                nobs1=sample_size,
                alpha=alpha,
                alternative='two-sided'
            )
            return {
                'effect_size': effect_size,
                'alpha': alpha,
                'sample_size': sample_size,
                'achieved_power': calculated_power
            }

    @staticmethod
    def comprehensive_model_comparison(y_true: np.ndarray,
                                     model_predictions: Dict[str, np.ndarray],
                                     reference_model: str = 'baseline',
                                     n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        全面的模型比较

        Args:
            y_true: 真实标签
            model_predictions: 模型预测字典
            reference_model: 参考模型名称
            n_bootstrap: Bootstrap次数

        Returns:
            pd.DataFrame: 比较结果表
        """
        results = []

        # 获取参考模型预测
        if reference_model not in model_predictions:
            raise ValueError(f"Reference model {reference_model} not found")

        ref_pred = model_predictions[reference_model]

        # 比较每个模型与参考模型
        for model_name, pred in model_predictions.items():
            if model_name == reference_model:
                continue

            # 计算基础指标
            auc_ref = roc_auc_score(y_true, ref_pred)
            auc_new = roc_auc_score(y_true, pred)
            auc_diff = auc_new - auc_ref

            # DeLong检验
            z_stat, p_delong, _ = StatisticalTests.delong_test(y_true, ref_pred, pred)

            # NRI计算
            nri_total, nri_events, nri_nonevents, nri_details = StatisticalTests.calculate_nri(
                y_true, ref_pred, pred
            )

            # IDI计算
            idi_total, idi_events, idi_details = StatisticalTests.calculate_idi(
                y_true, ref_pred, pred
            )

            # Bootstrap置信区间
            bootstrap_auc = StatisticalTests.bootstrap_test_statistic(
                y_true, ref_pred, pred, 'delong', n_bootstrap
            )

            results.append({
                'model': model_name,
                'reference': reference_model,
                'auc_ref': auc_ref,
                'auc_new': auc_new,
                'auc_diff': auc_diff,
                'delong_z': z_stat,
                'delong_p': p_delong,
                'nri_total': nri_total,
                'nri_events': nri_events,
                'nri_nonevents': nri_nonevents,
                'idi_total': idi_total,
                'idi_events': idi_events,
                'bootstrap_ci_lower': bootstrap_auc['ci_lower'],
                'bootstrap_ci_upper': bootstrap_auc['ci_upper'],
                'bootstrap_p': bootstrap_auc['p_value_bootstrap']
            })

        return pd.DataFrame(results)