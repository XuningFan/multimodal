"""
数据提取模块：从多中心Excel文件读取原始数据
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExcelExtractor:
    """Excel数据提取器，处理多级表头和多中心数据"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化提取器

        Args:
            config: 包含source_excel, sheets, binary_map等配置
        """
        self.source_path = Path(config['source_excel'])
        self.sheets = config['sheets']
        self.binary_map = config['binary_map']
        self.center_map = config['center_map']

    def extract_all_sheets(self) -> Dict[str, pd.DataFrame]:
        """
        提取所有工作表数据

        Returns:
            Dict[str, pd.DataFrame]: 中心名称到数据框的映射
        """
        logger.info(f"开始提取Excel数据: {self.source_path}")

        all_data = {}

        for sheet_name in self.sheets:
            logger.info(f"处理工作表: {sheet_name}")
            df = self._extract_single_sheet(sheet_name)
            all_data[sheet_name] = df

        return all_data

    def _extract_single_sheet(self, sheet_name: str) -> pd.DataFrame:
        """
        提取单个工作表

        Args:
            sheet_name: 工作表名称

        Returns:
            pd.DataFrame: 清理后的数据框
        """
        # 读取原始数据（保留多级表头）
        df_raw = pd.read_excel(
            self.source_path,
            sheet_name=sheet_name,
            header=[0, 1],  # 多级表头
            dtype=str
        )

        # 识别并移除汇总行
        df_clean = self._remove_summary_rows(df_raw)

        # 映射二元标记
        df_clean = self._map_binary_markers(df_clean)

        # 添加中心标识
        center_id = self.center_map.get(sheet_name, sheet_name)
        df_clean['center_id'] = center_id

        logger.info(f"工作表 {sheet_name} 提取完成，共 {len(df_clean)} 行数据")

        return df_clean

    def _remove_summary_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        识别并移除汇总行（通过正则表达式匹配）

        Args:
            df: 原始数据框

        Returns:
            pd.DataFrame: 移除汇总行后的数据框
        """
        # 常见的汇总行标识模式
        summary_patterns = [
            r'汇总|总计|合计|summary|total',
            r'平均|mean|average',
            r'标准差|std|std\.dev',
            r'中位数|median',
            r'最大值|max|maximum',
            r'最小值|min|minimum'
        ]

        combined_pattern = '|'.join(summary_patterns)

        # 检查第一列是否有汇总标识
        first_col = df.iloc[:, 0]
        mask = ~first_col.str.contains(combined_pattern, case=False, na=False)

        df_clean = df[mask].copy()

        removed_count = len(df) - len(df_clean)
        if removed_count > 0:
            logger.info(f"移除了 {removed_count} 行汇总数据")

        return df_clean

    def _map_binary_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        映射二元标记（√/×/0/空白）到数值

        Args:
            df: 原始数据框

        Returns:
            pd.DataFrame: 映射后的数据框
        """
        def map_value(val):
            if pd.isna(val) or val == '':
                return np.nan
            return self.binary_map.get(val, val)

        # 应用到所有字符串列
        df_mapped = df.copy()
        for col in df_mapped.select_dtypes(include=['object']).columns:
            df_mapped[col] = df_mapped[col].apply(map_value)

        return df_mapped

    def validate_extraction(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        验证提取结果

        Args:
            data_dict: 提取的数据字典

        Returns:
            Dict[str, Any]: 验证结果
        """
        validation_results = {
            'total_patients': 0,
            'centers': len(data_dict),
            'center_counts': {},
            'missing_sheets': []
        }

        for sheet_name, df in data_dict.items():
            count = len(df)
            validation_results['center_counts'][sheet_name] = count
            validation_results['total_patients'] += count

        # 检查是否有工作表缺失
        for expected_sheet in self.sheets:
            if expected_sheet not in data_dict:
                validation_results['missing_sheets'].append(expected_sheet)

        return validation_results