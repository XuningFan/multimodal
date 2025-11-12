"""
数据转换模块：标准化命名、单位统一、数据类型转换
"""

import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class DataTransformer:
    """数据转换器，实现数据标准化和特征工程"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化转换器

        Args:
            config: 包含命名规则、单位转换规则等配置
        """
        self.unit_rules = config.get('unit_rules', {})
        self.value_ranges = config.get('value_ranges', {})
        self.excel_date_fields = config.get('excel_date_fields', [])
        self.pid_key = config['pid_key']

    def transform_all_centers(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        转换所有中心的数据

        Args:
            data_dict: 中心数据字典

        Returns:
            pd.DataFrame: 合并并标准化后的数据
        """
        transformed_dfs = []

        for center_name, df in data_dict.items():
            logger.info(f"转换中心 {center_name} 的数据")
            df_transformed = self._transform_single_center(df, center_name)
            transformed_dfs.append(df_transformed)

        # 合并所有中心数据
        df_combined = pd.concat(transformed_dfs, ignore_index=True)
        logger.info(f"合并后总数据量: {len(df_combined)}")

        return df_combined

    def _transform_single_center(self, df: pd.DataFrame, center_name: str) -> pd.DataFrame:
        """
        转换单个中心的数据

        Args:
            df: 原始数据框
            center_name: 中心名称

        Returns:
            pd.DataFrame: 转换后的数据框
        """
        # 1. 扁平化多级表头
        df_flat = self._flatten_multilevel_columns(df)

        # 2. 生成患者ID
        df_flat = self._generate_patient_ids(df_flat, center_name)

        # 3. 日期字段转换
        df_flat = self._convert_date_fields(df_flat)

        # 4. 单位统一
        df_flat = self._standardize_units(df_flat)

        # 5. 数据类型转换
        df_flat = self._convert_data_types(df_flat)

        # 6. 派生特征
        df_flat = self._derive_features(df_flat)

        return df_flat

    def _flatten_multilevel_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        扁平化多级表头到标准命名格式

        命名规则：<domain>__<field>__[timepoint]
        """
        new_columns = []

        for col in df.columns:
            if isinstance(col, tuple):
                # 多级表头处理
                domain = col[0].strip() if col[0] else 'unknown'
                field = col[1].strip() if col[1] else 'unknown'

                # 识别时间点
                timepoint = self._extract_timepoint(field)

                # 标准化字段名
                field_clean = self._clean_field_name(field, timepoint)

                new_col_name = f"{domain}__{field_clean}__{timepoint}"
            else:
                # 单级表头处理
                new_col_name = str(col)

            new_columns.append(new_col_name)

        df.columns = new_columns
        return df

    def _extract_timepoint(self, field: str) -> str:
        """从字段名中提取时间点"""
        # 时间点映射规则
        time_mapping = {
            '术前': 'preop',
            '术后第1天': 'd1',
            '术后第2天': 'd2',
            '术后第3天': 'd3',
            '峰值': 'peak',
            '最低值': 'low',
            '术中': 'intraop',
            '手术': 'surg',
            'cpb': 'cpb',
            '术前': 'preop',
            '入院': 'admission'
        }

        for pattern, timepoint in time_mapping.items():
            if pattern in field:
                return timepoint

        return 'unknown'

    def _clean_field_name(self, field: str, timepoint: str) -> str:
        """清理字段名，移除时间点相关文字"""
        # 移除已识别的时间点文字
        patterns_to_remove = [
            r'术前|术后|术中|入院|第[1-9]天|峰值|最低值',
            r'[（(][^）)]*[）)]',  # 移除括号内容
            r'\s+',  # 移除多余空格
        ]

        cleaned = field
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned).strip()

        return cleaned if cleaned else 'value'

    def _generate_patient_ids(self, df: pd.DataFrame, center_name: str) -> pd.DataFrame:
        """
        生成全局唯一的患者ID

        Args:
            df: 数据框
            center_name: 中心名称

        Returns:
            pd.DataFrame: 添加了pid的数据框
        """
        center_id = df['center_id'].iloc[0] if 'center_id' in df.columns else center_name

        # 使用行索引生成哈希ID
        pids = []
        for idx in df.index:
            raw_str = f"{center_id}_{idx}"
            hash_obj = hashlib.md5(raw_str.encode())
            pid = hash_obj.hexdigest()[:16]  # 取前16位
            pids.append(pid)

        df[self.pid_key] = pids
        return df

    def _convert_date_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换Excel日期字段"""
        for date_field in self.excel_date_fields:
            if date_field in df.columns:
                # 保留原始值
                df[f"{date_field}_raw"] = df[date_field].copy()

                # 转换Excel序列号到日期
                df[date_field] = pd.to_datetime(df[date_field], errors='coerce')

                logger.info(f"转换日期字段: {date_field}")

        return df

    def _standardize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """单位标准化"""
        conversion_log = []

        for pattern, rule in self.unit_rules.items():
            # 匹配字段模式
            matched_cols = [col for col in df.columns if self._match_pattern(col, pattern)]

            for col in matched_cols:
                if col in df.columns:
                    df, converted = self._apply_unit_conversion(df, col, rule)
                    if converted:
                        conversion_log.append(f"{col}: {rule}")

        if conversion_log:
            logger.info(f"单位转换: {conversion_log}")

        return df

    def _match_pattern(self, col_name: str, pattern: str) -> bool:
        """检查列名是否匹配模式"""
        # 简单的通配符匹配
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(regex_pattern, col_name))

    def _apply_unit_conversion(self, df: pd.DataFrame, col: str, rule: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
        """
        应用单位转换规则

        Args:
            df: 数据框
            col: 列名
            rule: 转换规则

        Returns:
            Tuple[pd.DataFrame, bool]: 转换后的数据框和是否进行了转换
        """
        if 'factor' in rule:
            # 数值转换
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col] * rule['factor']
            return df, True

        return df, False

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据类型转换和清理"""
        # 转换数值类型
        for col in df.columns:
            # 尝试转换为数值类型
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='ignore')

        return df

    def _derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """派生新特征"""
        # 手术持续时间
        if 'surg__start_dt' in df.columns and 'surg__end_dt' in df.columns:
            df['surg__duration_min'] = (
                df['surg__end_dt'] - df['surg__start_dt']
            ).dt.total_seconds() / 60

        logger.info("派生特征计算完成")

        return df

    def validate_transformation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证转换结果

        Args:
            df: 转换后的数据框

        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'pid_unique_count': df[self.pid_key].nunique(),
            'missing_value_report': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'value_range_violations': []
        }

        # 检查数值范围
        for pattern, (min_val, max_val) in self.value_ranges.items():
            matched_cols = [col for col in df.columns if self._match_pattern(col, pattern)]

            for col in matched_cols:
                if col in df.columns and df[col].dtype in ['float64', 'int64']:
                    violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
                    if violations > 0:
                        validation_results['value_range_violations'].append({
                            'column': col,
                            'violations': int(violations),
                            'range': [min_val, max_val]
                        })

        return validation_results