"""
特征工程工具函数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


class FeatureUtils:
    """特征工程工具类"""

    @staticmethod
    def encode_categorical_features(df: pd.DataFrame,
                                   categorical_cols: List[str],
                                   encoding_method: str = 'onehot') -> Tuple[pd.DataFrame, Union[LabelEncoder, OneHotEncoder]]:
        """
        编码分类特征

        Args:
            df: 输入数据框
            categorical_cols: 分类特征列名列表
            encoding_method: 编码方法 ('onehot' 或 'label')

        Returns:
            Tuple[pd.DataFrame, encoder]: 编码后的数据框和编码器
        """
        df_encoded = df.copy()

        if encoding_method == 'onehot':
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

            for col in categorical_cols:
                if col in df_encoded.columns:
                    # 执行独热编码
                    encoded_data = encoder.fit_transform(df_encoded[[col]])

                    # 创建新列名
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_encoded.index)

                    # 合并到原数据框
                    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                    df_encoded = df_encoded.drop(columns=[col])

        elif encoding_method == 'label':
            encoder_dict = {}

            for col in categorical_cols:
                if col in df_encoded.columns:
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                    encoder_dict[col] = encoder

            encoder = encoder_dict

        else:
            raise ValueError(f"Unsupported encoding method: {encoding_method}")

        logger.info(f"分类特征编码完成，方法: {encoding_method}")
        return df_encoded, encoder

    @staticmethod
    def remove_low_variance_features(df: pd.DataFrame,
                                    threshold: float = 0.01,
                                    exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, VarianceThreshold]:
        """
        移除低方差特征

        Args:
            df: 输入数据框
            threshold: 方差阈值
            exclude_cols: 排除的列名列表

        Returns:
            Tuple[pd.DataFrame, VarianceThreshold]: 处理后的数据框和方差阈值选择器
        """
        if exclude_cols is None:
            exclude_cols = []

        # 分离数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(feature_cols) == 0:
            logger.warning("没有找到数值特征进行方差筛选")
            return df.copy(), None

        # 方差阈值选择
        selector = VarianceThreshold(threshold=threshold)

        try:
            # 选择特征
            selected_features = selector.fit_transform(df[feature_cols])
            selected_feature_names = np.array(feature_cols)[selector.get_support()]

            # 构建新的数据框
            df_result = df[exclude_cols + selected_feature_names.tolist()].copy()

            logger.info(f"低方差特征移除: {len(feature_cols)} -> {len(selected_feature_names)}")
            return df_result, selector

        except Exception as e:
            logger.error(f"低方差特征移除失败: {e}")
            return df.copy(), None

    @staticmethod
    def handle_outliers(df: pd.DataFrame,
                       method: str = 'iqr',
                       exclude_cols: Optional[List[str]] = None,
                       action: str = 'clip') -> pd.DataFrame:
        """
        处理异常值

        Args:
            df: 输入数据框
            method: 检测方法 ('iqr', 'zscore', 'modified_zscore')
            exclude_cols: 排除的列名列表
            action: 处理方式 ('clip', 'remove', 'transform')

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        if exclude_cols is None:
            exclude_cols = ['pid', 'center_id']

        df_result = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        for col in numeric_cols:
            if method == 'iqr':
                lower_bound, upper_bound = FeatureUtils._detect_outliers_iqr(df_result[col])
            elif method == 'zscore':
                lower_bound, upper_bound = FeatureUtils._detect_outliers_zscore(df_result[col])
            elif method == 'modified_zscore':
                lower_bound, upper_bound = FeatureUtils._detect_outliers_modified_zscore(df_result[col])
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")

            if action == 'clip':
                df_result[col] = df_result[col].clip(lower_bound, upper_bound)
            elif action == 'remove':
                mask = (df_result[col] >= lower_bound) & (df_result[col] <= upper_bound)
                df_result = df_result[mask]
            elif action == 'transform':
                # 对数变换
                if (df_result[col] > 0).all():
                    df_result[col] = np.log1p(df_result[col])
                else:
                    logger.warning(f"列 {col} 包含非正值，跳过对数变换")

        logger.info(f"异常值处理完成，方法: {method}, 动作: {action}")
        return df_result

    @staticmethod
    def _detect_outliers_iqr(series: pd.Series) -> Tuple[float, float]:
        """IQR方法检测异常值"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    @staticmethod
    def _detect_outliers_zscore(series: pd.Series) -> Tuple[float, float]:
        """Z-score方法检测异常值"""
        mean = series.mean()
        std = series.std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        return lower_bound, upper_bound

    @staticmethod
    def _detect_outliers_modified_zscore(series: pd.Series) -> Tuple[float, float]:
        """修正Z-score方法检测异常值"""
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        threshold = 3.5

        outlier_mask = np.abs(modified_z_scores) > threshold
        lower_bound = series[~outlier_mask].min()
        upper_bound = series[~outlier_mask].max()
        return lower_bound, upper_bound

    @staticmethod
    def create_interaction_features(df: pd.DataFrame,
                                   feature_pairs: Optional[List[Tuple[str, str]]] = None,
                                   max_features: int = 100) -> pd.DataFrame:
        """
        创建交互特征

        Args:
            df: 输入数据框
            feature_pairs: 指定的特征对列表
            max_features: 最大交互特征数量

        Returns:
            pd.DataFrame: 包含交互特征的数据框
        """
        df_result = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 移除ID列
        numeric_cols = [col for col in numeric_cols if col not in ['pid', 'center_id']]

        if feature_pairs is None:
            # 自动选择特征对（基于相关性）
            feature_pairs = FeatureUtils._select_feature_pairs(df[numeric_cols], max_features)

        interaction_count = 0
        for col1, col2 in feature_pairs:
            if col1 in df_result.columns and col2 in df_result.columns:
                # 乘积交互
                interaction_col = f"interaction_{col1}_x_{col2}"
                df_result[interaction_col] = df_result[col1] * df_result[col2]
                interaction_count += 1

                if interaction_count >= max_features:
                    break

        logger.info(f"创建了 {interaction_count} 个交互特征")
        return df_result

    @staticmethod
    def _select_feature_pairs(df: pd.DataFrame, max_pairs: int) -> List[Tuple[str, str]]:
        """选择特征对进行交互"""
        # 计算相关性矩阵
        corr_matrix = df.corr().abs()

        # 获取上三角矩阵的相关性
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # 找到中等相关性的特征对（既不高相关也不低相关）
        feature_pairs = []
        for i in range(len(upper_tri.columns)):
            for j in range(i):
                corr_val = upper_tri.iloc[j, i]
                if 0.1 < corr_val < 0.8:  # 中等相关性
                    feature_pairs.append((upper_tri.columns[j], upper_tri.columns[i]))

        # 按相关性排序并选择前N对
        feature_pairs.sort(key=lambda pair: corr_matrix.loc[pair[0], pair[1]], reverse=True)

        return feature_pairs[:max_pairs]

    @staticmethod
    def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        获取特征分组

        Args:
            df: 输入数据框

        Returns:
            Dict[str, List[str]]: 特征分组字典
        """
        feature_groups = {
            'demographics': [col for col in df.columns if col.startswith('demo__')],
            'medical_history': [col for col in df.columns if col.startswith('mhx__')],
            'surgery': [col for col in df.columns if col.startswith('surg__')],
            'cpb': [col for col in df.columns if col.startswith('cpb__')],
            'perfusion': [col for col in df.columns if col.startswith('perf__')],
            'transfusion': [col for col in df.columns if col.startswith('transfusion__')],
            'labs_preop': [col for col in df.columns if col.startswith('lab__preop')],
            'imaging': [col for col in df.columns if col.startswith('img__')],
            'radiomics': [col for col in df.columns if col.startswith('rad__')],
            'derived': [col for col in df.columns if col.startswith('derived__')],
            'interaction': [col for col in df.columns if col.startswith('interaction_')]
        }

        # 移除空分组
        feature_groups = {k: v for k, v in feature_groups.items() if v}

        return feature_groups

    @staticmethod
    def generate_feature_importance_report(feature_importance: Dict[str, float],
                                         feature_groups: Dict[str, List[str]],
                                         top_n: int = 20) -> pd.DataFrame:
        """
        生成特征重要性报告

        Args:
            feature_importance: 特征重要性字典
            feature_groups: 特征分组
            top_n: 显示的前N个特征

        Returns:
            pd.DataFrame: 特征重要性报告
        """
        # 转换为数据框
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['feature', 'importance']
        )

        # 排序
        importance_df = importance_df.sort_values('importance', ascending=False)

        # 添加分组信息
        def get_feature_group(feature):
            for group, features in feature_groups.items():
                if feature in features:
                    return group
            return 'other'

        importance_df['group'] = importance_df['feature'].apply(get_feature_group)

        # 添加排名
        importance_df['rank'] = range(1, len(importance_df) + 1)

        return importance_df.head(top_n)