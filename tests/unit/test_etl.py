"""
ETL模块单元测试
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from etl.extract import ExcelExtractor
from etl.transform import DataTransformer
from etl.validators import LeakageGuard

class TestExcelExtraction(unittest.TestCase):
    """Excel数据提取测试"""

    def setUp(self):
        """测试前准备"""
        self.config = {
            'source_excel': 'test_data.xlsx',
            'sheets': ['CenterA', 'CenterB'],
            'binary_map': {"√": 1, "×": 0, "0": 0, "": None},
            'center_map': {'CenterA': 'C1', 'CenterB': 'C2'}
        }
        self.extractor = ExcelExtractor(self.config)

    def test_binary_mapping(self):
        """测试二元标记映射"""
        test_cases = [
            ("√", 1),
            ("×", 0),
            ("0", 0),
            ("", None),
            (" ", None)
        ]

        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = self.extractor._map_binary_markers(pd.DataFrame({'A': [input_val]}))
                self.assertEqual(result.iloc[0, 'A'], expected)

    def test_excel_date_conversion(self):
        """测试Excel日期转换"""
        # Excel序列号 45797 = 2025-05-20
        excel_serial = 45797
        expected_date = datetime(2025, 5, 20)

        # 测试日期转换逻辑
        converted_date = pd.to_datetime(excel_serial, origin='1899-12-30', unit='D')
        self.assertEqual(converted_date.date(), expected_date.date())

    def test_pid_generation(self):
        """测试患者ID生成"""
        df = pd.DataFrame({'center_id': ['C1', 'C1', 'C2']})
        result_df = self.extractor._generate_patient_ids(df, 'C1')

        # 检查ID生成
        self.assertTrue(all(isinstance(pid, str) for pid in result_df['pid']))
        self.assertEqual(len(result_df['pid'].unique()), len(df))

    def test_summary_row_removal(self):
        """测试汇总行移除"""
        # 创建包含汇总行的测试数据
        df = pd.DataFrame({
            'A': ['Patient1', 'Patient2', '汇总', '总计'],
            'B': [1, 2, 3, 5]
        })

        result_df = self.extractor._remove_summary_rows(df)

        # 应该移除汇总行
        self.assertEqual(len(result_df), 2)
        self.assertNotIn('汇总', result_df['A'].values)
        self.assertNotIn('总计', result_df['A'].values)


class TestDataTransformation(unittest.TestCase):
    """数据转换测试"""

    def setUp(self):
        """测试前准备"""
        self.config = {
            'value_ranges': {
                'lab__Cr_*': [20, 1500],
                'demo__age_yr': [18, 100]
            },
            'excel_date_fields': ['surg__start_dt_raw'],
            'pid_key': 'pid'
        }
        self.transformer = DataTransformer(self.config)

    def test_multilevel_column_flattening(self):
        """测试多级表头扁平化"""
        # 创建多级表头数据框
        columns = pd.MultiIndex.from_tuples([
            ('demographics', 'age', 'preop'),
            ('laboratory', 'creatinine', 'preop')
        ])
        df = pd.DataFrame([[25, 80]], columns=columns)

        result_df = self.transformer._flatten_multilevel_columns(df)

        # 检查列名扁平化
        expected_columns = ['demo__age__preop', 'lab__creatinine__preop']
        self.assertEqual(list(result_df.columns), expected_columns)

    def test_timepoint_extraction(self):
        """测试时间点提取"""
        test_cases = [
            ("术前第1天", "d1"),
            ("术后第2天", "d2"),
            ("峰值", "peak"),
            ("最低值", "low"),
            ("术中", "intraop")
        ]

        for field_name, expected in test_cases:
            with self.subTest(field_name=field_name):
                result = self.transformer._extract_timepoint(field_name)
                self.assertEqual(result, expected)

    def test_value_range_validation(self):
        """测试数值范围验证"""
        # 创建测试数据，包含正常值和异常值
        df = pd.DataFrame({
            'lab__Cr__preop': [80, 1500, 2000, 50],  # 2000是异常值
            'demo__age_yr': [25, 65, 150, 30]       # 150是异常值
        })

        validation_results = self.transformer.validate_transformation(df)

        # 检查是否检测到异常值
        violations = validation_results['value_range_violations']
        self.assertTrue(len(violations) > 0)

        # 检查违规列
        violation_columns = [v['column'] for v in violations]
        self.assertIn('lab__Cr__preop', violation_columns)
        self.assertIn('demo__age_yr', violation_columns)


class TestLeakageGuard(unittest.TestCase):
    """特征泄漏守卫测试"""

    def setUp(self):
        """测试前准备"""
        self.config = {
            'blacklist_features': [
                'lab__*_d1',
                'lab__*_d2',
                'lab__*_peak'
            ],
            'allowed_domains': ['demo__', 'surg__', 'cpb__']
        }
        self.guard = LeakageGuard(self.config)

    def test_feature_blacklisting(self):
        """测试特征黑名单检查"""
        blacklisted_features = [
            'lab__Cr__d1',
            'lab__ALT__d2',
            'lab__TBil__peak'
        ]

        for feature in blacklisted_features:
            with self.subTest(feature=feature):
                self.assertTrue(self.guard._is_blacklisted(feature))

    def test_safe_features(self):
        """测试安全特征检查"""
        safe_features = [
            'demo__age_yr',
            'surg__duration_min',
            'cpb__time_min',
            'lab__Cr__preop'  # 术前化验是安全的
        ]

        for feature in safe_features:
            with self.subTest(feature=feature):
                self.assertFalse(self.guard._is_blacklisted(feature))

    def test_training_features_guard(self):
        """测试训练特征守卫"""
        mixed_features = [
            'demo__age_yr',      # 安全
            'lab__Cr__preop',    # 安全
            'lab__Cr__d1',       # 黑名单
            'surg__duration_min' # 安全
        ]

        guard_result = self.guard.guard_training_features(mixed_features)

        # 应该检测到黑名单特征
        self.assertFalse(guard_result['is_safe'])
        self.assertIn('lab__Cr__d1', guard_result['blacklisted_features'])
        self.assertEqual(len(guard_result['safe_features']), 3)  # 移除了黑名单特征


if __name__ == '__main__':
    unittest.main()