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
        first_sheet_columns = None

        for sheet_name in self.sheets:
            logger.info(f"处理工作表: {sheet_name}")
            df = self._extract_single_sheet(sheet_name)

            # 检查列名一致性
            if first_sheet_columns is None:
                first_sheet_columns = df.columns.tolist()
                logger.info(f"第一个工作表 '{sheet_name}' 的列结构已设置为基准，共 {len(first_sheet_columns)} 列")
            else:
                current_columns = df.columns.tolist()
                if current_columns != first_sheet_columns:
                    logger.error(f"工作表 '{sheet_name}' 的列结构与第一个工作表不一致!")
                    logger.error(f"第一个工作表列数: {len(first_sheet_columns)}")
                    logger.error(f"当前工作表列数: {len(current_columns)}")

                    # 找出差异列
                    missing_cols = set(first_sheet_columns) - set(current_columns)
                    extra_cols = set(current_columns) - set(first_sheet_columns)

                    if missing_cols:
                        logger.error(f"缺失的列: {list(missing_cols)}")
                    if extra_cols:
                        logger.error(f"多余的列: {list(extra_cols)}")

                    raise ValueError(f"工作表 '{sheet_name}' 列结构不一致，请检查Excel文件")

            all_data[sheet_name] = df

        logger.info(f"所有工作表列名一致性检查通过")
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

        # 映射二元标记
        df_clean = self._map_binary_markers(df_raw)

        # 添加中心标识
        center_id = self.center_map.get(sheet_name, sheet_name)
        df_clean['center_id'] = center_id

        logger.info(f"工作表 {sheet_name} 提取完成，共 {len(df_clean)} 行数据")

        return df_clean

    
    def _map_binary_markers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        映射二元标记（√/×/0/空白）到数值

        规则：
        1. 如果列中包含√，该列进行二元转换：√标记为1，所有非√标记（×、0、空白等）为0
        2. 其他列按原binary_map转换，空白表示缺失

        Args:
            df: 原始数据框

        Returns:
            pd.DataFrame: 映射后的数据框
        """
        df_mapped = df.copy()

        # 检查哪些列包含√标记
        checkmark_cols = []
        for col in df_mapped.select_dtypes(include=['object']).columns:
            if df_mapped[col].astype(str).str.contains('√').any():
                checkmark_cols.append(col)

        # 处理包含√标记的列
        for col in checkmark_cols:
            df_mapped[col] = df_mapped[col].apply(lambda x: 1 if str(x) == '√' else 0)

        # 处理其他列（不包含√标记）
        other_cols = [col for col in df_mapped.select_dtypes(include=['object']).columns if col not in checkmark_cols]
        for col in other_cols:
            def map_value(val):
                if pd.isna(val) or val == '':
                    return np.nan  # 空白表示缺失
                return self.binary_map.get(val, val)

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