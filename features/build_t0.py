"""
T0基线特征构建模块：构建术前+术中基线特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)


class T0FeatureBuilder:
    """T0基线特征构建器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征构建器

        Args:
            config: 包含特征构建规则的配置
        """
        self.config = config
        self.t0_domains = config.get('t0_domains', [
            'demo__', 'mhx__', 'surg__', 'cpb__', 'perf__', 'transfusion__',
            'lab__preop'
        ])
        self.derived_features = config.get('derived_features', {})
        self.scaler_config = config.get('scaler', {})

    def build_t0_features(self, patients_df: pd.DataFrame,
                         surgery_cpb_df: pd.DataFrame,
                         labs_long_df: pd.DataFrame,
                         imaging_meta_df: pd.DataFrame) -> pd.DataFrame:
        """
        构建T0特征集

        Args:
            patients_df: 患者基础信息表
            surgery_cpb_df: 手术CPB信息表
            labs_long_df: 化验长表
            imaging_meta_df: 影像元数据表

        Returns:
            pd.DataFrame: T0特征数据框
        """
        logger.info("开始构建T0特征集")

        # 1. 合并基础表
        t0_df = self._merge_base_tables(patients_df, surgery_cpb_df, imaging_meta_df)

        # 2. 添加术前化验特征
        t0_df = self._add_preop_labs(t0_df, labs_long_df)

        # 3. 创建派生特征
        t0_df = self._create_derived_features(t0_df)

        # 4. 特征选择和清理
        t0_df = self._select_and_clean_features(t0_df)

        # 5. 数据预处理
        t0_df = self._preprocess_features(t0_df)

        logger.info(f"T0特征构建完成: {len(t0_df)} 行, {len(t0_df.columns)} 特征")
        return t0_df

    def _merge_base_tables(self, patients_df: pd.DataFrame,
                          surgery_cpb_df: pd.DataFrame,
                          imaging_meta_df: pd.DataFrame) -> pd.DataFrame:
        """合并基础表"""
        # 从患者表开始
        t0_df = patients_df.copy()

        # 合并手术CPB信息
        if 'pid' in surgery_cpb_df.columns:
            t0_df = t0_df.merge(surgery_cpb_df, on='pid', how='left')

        # 合并影像信息
        if 'pid' in imaging_meta_df.columns:
            t0_df = t0_df.merge(imaging_meta_df, on='pid', how='left')

        logger.info(f"基础表合并后: {len(t0_df)} 行, {len(t0_df.columns)} 列")
        return t0_df

    def _add_preop_labs(self, t0_df: pd.DataFrame, labs_long_df: pd.DataFrame) -> pd.DataFrame:
        """添加术前化验特征"""
        # 筛选术前化验
        preop_labs = labs_long_df[labs_long_df['timepoint'] == 'preop'].copy()

        # 透视表格：每个化验项作为一列
        preop_labs_pivot = preop_labs.pivot_table(
            index='pid',
            columns='analyte',
            values='value',
            aggfunc='first'
        ).reset_index()

        # 重命名列
        preop_labs_pivot.columns = ['pid'] + [
            f'lab__{col}__preop' for col in preop_labs_pivot.columns[1:]
        ]

        # 合并到T0数据框
        if 'pid' in t0_df.columns:
            t0_df = t0_df.merge(preop_labs_pivot, on='pid', how='left')

        logger.info(f"添加术前化验后: {len(t0_df)} 行, {len(t0_df.columns)} 列")
        return t0_df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建派生特征"""
        df_derived = df.copy()

        # 1. 连续特征衍生
        df_derived = self._create_continuous_derived_features(df_derived)

        # 2. 交互特征
        df_derived = self._create_interaction_features(df_derived)

        # 3. 阈值特征
        df_derived = self._create_threshold_features(df_derived)

        # 4. 时间归一化特征
        df_derived = self._create_time_normalized_features(df_derived)

        logger.info(f"派生特征创建后: {len(df_derived)} 行, {len(df_derived)} 列")
        return df_derived

    def _create_continuous_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建连续特征衍生"""
        # 体重指数（如果有身高体重）
        if 'demo__height_cm' in df.columns and 'demo__weight_kg' in df.columns:
            df['demo__bmi'] = df['demo__weight_kg'] / (df['demo__height_cm']/100)**2

        # 体表面积
        if 'demo__height_cm' in df.columns and 'demo__weight_kg' in df.columns:
            df['demo__bsa'] = 0.007184 * (df['demo__weight_kg']**0.425) * (df['demo__height_cm']**0.725)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互特征"""
        # CPB时间 × 最低温度
        if 'cpb__time_min' in df.columns and 'cpb__nadir_temp_c' in df.columns:
            df['derived__cpb_time_temp_interaction'] = (
                df['cpb__time_min'] * df['cpb__nadir_temp_c']
            )

        # 阻断时间 × CPB时间比率
        if 'cpb__crossclamp_min' in df.columns and 'cpb__time_min' in df.columns:
            df['derived__crossclamp_ratio'] = (
                df['cpb__crossclamp_min'] / df['cpb__time_min'].replace(0, np.nan)
            )

        return df

    def _create_threshold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建阈值特征"""
        # 低温阈值
        if 'cpb__nadir_temp_c' in df.columns:
            df['derived__hypothermia_severe'] = (df['cpb__nadir_temp_c'] <= 26).astype(int)
            df['derived__hypothermia_moderate'] = (
                (df['cpb__nadir_temp_c'] > 26) & (df['cpb__nadir_temp_c'] <= 28)
            ).astype(int)

        # 长时间CPB阈值
        if 'cpb__time_min' in df.columns:
            df['derived__cpb_prolonged'] = (df['cpb__time_min'] > 180).astype(int)

        # 术前肾功能异常
        if 'lab__Cr__preop' in df.columns:
            df['derived__preop_renal_impairment'] = (df['lab__Cr__preop'] > 133).astype(int)

        return df

    def _create_time_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间归一化特征"""
        # 输血率（U/min）
        if 'transfusion__rbc_u' in df.columns and 'cpb__time_min' in df.columns:
            df['derived__rbc_rate_per_min'] = (
                df['transfusion__rbc_u'] / df['cpb__time_min'].replace(0, np.nan)
            )

        return df

    def _select_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征选择和清理"""
        # 1. 选择T0允许的域
        t0_cols = ['pid', 'center_id']  # 保留ID列

        for col in df.columns:
            # 检查是否属于允许的域
            domain_found = False
            for domain in self.t0_domains:
                if col.startswith(domain):
                    t0_cols.append(col)
                    domain_found = True
                    break
            # 包含派生特征
            if not domain_found and col.startswith('derived__'):
                t0_cols.append(col)

        # 选择列
        df_selected = df[t0_cols].copy()

        # 2. 移除常数列
        constant_cols = []
        for col in df_selected.columns:
            if col not in ['pid', 'center_id']:
                if df_selected[col].nunique() <= 1:
                    constant_cols.append(col)

        if constant_cols:
            df_selected = df_selected.drop(columns=constant_cols)
            logger.info(f"移除常数列: {constant_cols}")

        # 3. 移除高缺失率列
        high_missing_cols = []
        for col in df_selected.columns:
            if col not in ['pid', 'center_id']:
                missing_rate = df_selected[col].isnull().sum() / len(df_selected)
                if missing_rate > 0.5:  # 缺失率>50%
                    high_missing_cols.append(col)

        if high_missing_cols:
            df_selected = df_selected.drop(columns=high_missing_cols)
            logger.info(f"移除高缺失率列: {high_missing_cols}")

        return df_selected

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df_processed = df.copy()

        # 1. 缺失值处理
        df_processed = self._handle_missing_values(df_processed)

        # 2. 数值特征标准化
        df_processed = self._scale_features(df_processed)

        return df_processed

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 数值列使用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['pid']]

        if numeric_cols:
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # 分类列使用众数填充
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['pid', 'center_id']]

        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])

        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """数值特征标准化"""
        scaler_type = self.scaler_config.get('type', 'standard')

        # 选择数值列（排除ID和二元特征）
        numeric_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['pid'] and not col.endswith('_severe') and not col.endswith('_moderate'):
                numeric_cols.append(col)

        if numeric_cols and scaler_type != 'none':
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                return df

            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

            # 保存scaler用于后续推理
            self.fitted_scaler = scaler

        return df

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """获取特征重要性分组"""
        return {
            'demographics': [col for col in self.get_feature_names() if col.startswith('demo__')],
            'medical_history': [col for col in self.get_feature_names() if col.startswith('mhx__')],
            'surgery_cpb': [col for col in self.get_feature_names() if col.startswith(('surg__', 'cpb__'))],
            'preop_labs': [col for col in self.get_feature_names() if col.startswith('lab__preop')],
            'derived': [col for col in self.get_feature_names() if col.startswith('derived__')]
        }

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表（需要先运行build_t0_features）"""
        if hasattr(self, 'last_features'):
            return self.last_features
        return []