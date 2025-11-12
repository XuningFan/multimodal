"""
数据验证模块：数据质量检查和防泄漏守卫
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化验证器

        Args:
            config: 包含验证规则的配置
        """
        self.value_ranges = config.get('value_ranges', {})
        self.blacklist_features = config.get('blacklist_features', [])
        self.required_features = config.get('required_features', [])

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        全面验证数据集

        Args:
            df: 待验证的数据框

        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': self._check_missing_values(df),
            'value_range_violations': self._check_value_ranges(df),
            'duplicate_records': self._check_duplicates(df),
            'feature_leakage_check': self._check_feature_leakage(df),
            'data_consistency': self._check_data_consistency(df),
            'is_valid': True
        }

        # 判断整体是否有效
        if (validation_results['value_range_violations'] or
            validation_results['duplicate_records'] > 0 or
            validation_results['feature_leakage_check']['has_leakage']):
            validation_results['is_valid'] = False

        return validation_results

    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查缺失值"""
        missing_summary = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_rate = missing_count / len(df)

            if missing_count > 0:
                missing_summary[col] = {
                    'count': int(missing_count),
                    'rate': float(missing_rate)
                }

        return missing_summary

    def _check_value_ranges(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """检查数值范围违规"""
        violations = []

        for pattern, (min_val, max_val) in self.value_ranges.items():
            matched_cols = self._match_columns(df.columns, pattern)

            for col in matched_cols:
                if col in df.columns and df[col].dtype in ['float64', 'int64']:
                    # 计算违规数量
                    invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                    invalid_count = invalid_mask.sum()

                    if invalid_count > 0:
                        violations.append({
                            'column': col,
                            'violations': int(invalid_count),
                            'range': [min_val, max_val],
                            'violation_rate': float(invalid_count / len(df))
                        })

        return violations

    def _check_duplicates(self, df: pd.DataFrame) -> int:
        """检查重复记录"""
        if 'pid' in df.columns:
            return int(df['pid'].duplicated().sum())
        else:
            return int(df.duplicated().sum())

    def _check_feature_leakage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查特征泄漏（使用术后信息预测术后结局）"""
        leakage_result = {
            'has_leakage': False,
            'blacklisted_features_found': [],
            'postoperative_features': []
        }

        # 检查黑名单特征
        for col in df.columns:
            for blacklist_pattern in self.blacklist_features:
                if self._match_pattern(col, blacklist_pattern):
                    leakage_result['has_leakage'] = True
                    leakage_result['blacklisted_features_found'].append(col)

        # 检查其他可能的术后特征
        postop_patterns = [
            r'.*_d[1-9]$',  # 术后第X天
            r'.*_postop$',  # 术后
            r'.*_peak$',    # 峰值
            r'.*_nadir$'    # 最低值
        ]

        for col in df.columns:
            for pattern in postop_patterns:
                if re.match(pattern, col) and col.startswith(('lab__', 'vitals__')):
                    leakage_result['postoperative_features'].append(col)

        return leakage_result

    def _check_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据一致性"""
        consistency_checks = {}

        # 检查时间顺序（如果有时间字段）
        time_fields = [col for col in df.columns if 'dt' in col.lower() or 'time' in col.lower()]
        for field in time_fields:
            if field in df.columns:
                consistency_checks[f'{field}_consistency'] = self._check_time_consistency(df, field)

        # 检查逻辑一致性
        if 'surg__duration_min' in df.columns and 'cpb__time_min' in df.columns:
            consistency_checks['cpb_vs_surgery_time'] = self._check_surgery_time_logic(df)

        return consistency_checks

    def _match_columns(self, columns: List[str], pattern: str) -> List[str]:
        """匹配列名模式"""
        matched = []
        regex_pattern = pattern.replace('*', '.*')
        for col in columns:
            if re.match(regex_pattern, col):
                matched.append(col)
        return matched

    def _match_pattern(self, name: str, pattern: str) -> bool:
        """检查单个名称是否匹配模式"""
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(regex_pattern, name))

    def _check_time_consistency(self, df: pd.DataFrame, time_field: str) -> Dict[str, Any]:
        """检查时间字段一致性"""
        if time_field not in df.columns:
            return {'status': 'field_missing'}

        # 检查是否为日期时间类型
        if not pd.api.types.is_datetime64_any_dtype(df[time_field]):
            try:
                pd.to_datetime(df[time_field])
                return {'status': 'convertible'}
            except:
                return {'status': 'invalid_format'}

        return {'status': 'valid'}

    def _check_surgery_time_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查手术时间和CPB时间的逻辑关系"""
        if 'surg__duration_min' not in df.columns or 'cpb__time_min' not in df.columns:
            return {'status': 'fields_missing'}

        # CPB时间不应该超过手术时间
        mask = df['cpb__time_min'] > df['surg__duration_min']
        violations = mask.sum()

        return {
            'status': 'valid' if violations == 0 else 'violations',
            'violations': int(violations),
            'violation_rate': float(violations / len(df))
        }


class LeakageGuard:
    """特征泄漏守卫"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化守卫

        Args:
            config: 包含黑名单和规则的配置
        """
        self.blacklist_patterns = config.get('blacklist_features', [])
        self.t0_boundary = config.get('t0_boundary', '术毕')
        self.allowed_domains = config.get('allowed_domains', [
            'demo__', 'mhx__', 'surg__', 'cpb__', 'perf__', 'transfusion__'
        ])

    def guard_training_features(self, features: List[str]) -> Dict[str, Any]:
        """
        守卫训练特征，确保没有特征泄漏

        Args:
            features: 特征列表

        Returns:
            Dict[str, Any]: 守卫结果
        """
        guard_result = {
            'is_safe': True,
            'blacklisted_features': [],
            'suspicious_features': [],
            'safe_features': features.copy()
        }

        for feature in features:
            # 检查黑名单
            if self._is_blacklisted(feature):
                guard_result['is_safe'] = False
                guard_result['blacklisted_features'].append(feature)
                guard_result['safe_features'].remove(feature)
                continue

            # 检查可疑特征
            if self._is_suspicious(feature):
                guard_result['suspicious_features'].append(feature)

        if guard_result['blacklisted_features']:
            logger.error(f"发现黑名单特征（泄漏风险）: {guard_result['blacklisted_features']}")

        if guard_result['suspicious_features']:
            logger.warning(f"发现可疑特征（请手动检查）: {guard_result['suspicious_features']}")

        return guard_result

    def _is_blacklisted(self, feature: str) -> bool:
        """检查特征是否在黑名单中"""
        for pattern in self.blacklist_patterns:
            if self._match_pattern(feature, pattern):
                return True
        return False

    def _is_suspicious(self, feature: str) -> bool:
        """检查特征是否可疑"""
        suspicious_patterns = [
            r'.*_d[1-9]$',      # 术后第X天
            r'.*_postop$',      # 术后
            r'.*_peak$',        # 峰值
            r'.*_nadir$',       # 最低值
            r'complication__',  # 并发症相关
            r'outcome__'        # 结局相关
        ]

        for pattern in suspicious_patterns:
            if self._match_pattern(feature, pattern):
                return True

        return False

    def _match_pattern(self, name: str, pattern: str) -> bool:
        """模式匹配"""
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(regex_pattern, name))


def guard_no_postop_features(df: pd.DataFrame, config: Dict[str, Any]) -> bool:
    """
    快速守卫函数：确保数据框中没有术后特征

    Args:
        df: 数据框
        config: 配置信息

    Returns:
        bool: 是否安全（True=无术后特征）
    """
    guard = LeakageGuard(config)
    result = guard.guard_training_features(df.columns.tolist())
    return result['is_safe']