"""
放射组学特征提取模块：从CT影像中提取传统放射组学特征
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from radiomics import featureextractor
import pickle

logger = logging.getLogger(__name__)


class RadiomicsExtractor:
    """放射组学特征提取器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化提取器

        Args:
            config: 配置信息
        """
        self.config = config
        self.feature_classes = config.get('feature_classes', [
            'firstorder', 'shape', 'texture', 'glcm', 'glrlm', 'glszm'
        ])
        self.resampling_spacing = config.get('resampling_spacing', [1.0, 1.0, 1.0])
        self.bin_width = config.get('bin_width', 25)
        self.center_correction = config.get('center_correction', True)

        # 初始化特征提取器
        self.extractor = self._setup_extractor()

    def _setup_extractor(self) -> featureextractor.RadiomicsFeatureExtractor:
        """设置放射组学特征提取器"""
        extractor = featureextractor.RadiomicsFeatureExtractor()

        # 设置参数
        settings = {
            'binWidth': self.bin_width,
            'interpolator': 'sitkBSpline',
            'resampledPixelSpacing': self.resampling_spacing,
            'normalize': True,
            'normalizeScale': 100
        }

        for setting, value in settings.items():
            extractor.settings[setting] = value

        # 启用特征类
        enabled_features = {}
        for feature_class in self.feature_classes:
            enabled_features[feature_class] = []

        extractor.enabledFeatures = enabled_features

        return extractor

    def extract_radiomics_features(self, imaging_meta_df: pd.DataFrame,
                                   mask_dir: Optional[str] = None) -> pd.DataFrame:
        """
        提取放射组学特征

        Args:
            imaging_meta_df: 影像元数据表
            mask_dir: 掩码目录路径

        Returns:
            pd.DataFrame: 包含放射组学特征的结果表
        """
        logger.info("开始提取放射组学特征")

        # 筛选有影像的患者
        img_available_df = imaging_meta_df[
            imaging_meta_df['img__cta_preop_available'] == 1
        ].copy()

        if len(img_available_df) == 0:
            logger.warning("没有可用的影像数据进行放射组学提取")
            return imaging_meta_df.copy()

        # 提取特征
        radiomics_data = []
        failed_cases = []

        for idx, row in img_available_df.iterrows():
            pid = row['pid']
            image_path = row.get('img__path', f'data/cta_images/{pid}.nii.gz')

            try:
                # 尝试找到对应的掩码
                mask_path = self._find_mask_path(pid, mask_dir)

                if mask_path and Path(mask_path).exists():
                    # 有掩码，使用掩码提取
                    features = self._extract_with_mask(image_path, mask_path)
                else:
                    # 没有掩码，使用整个影像
                    features = self._extract_without_mask(image_path)

                features['pid'] = pid
                radiomics_data.append(features)

            except Exception as e:
                logger.error(f"提取放射组学特征失败 {pid}: {e}")
                failed_cases.append({'pid': pid, 'error': str(e)})

        # 转换为数据框
        if radiomics_data:
            radiomics_df = pd.DataFrame(radiomics_data)

            # 与原数据合并
            result_df = img_available_df.merge(radiomics_df, on='pid', how='left')
        else:
            logger.error("所有放射组学特征提取都失败了")
            result_df = img_available_df.copy()

        # 记录失败案例
        if failed_cases:
            logger.warning(f"放射组学提取失败案例数: {len(failed_cases)}")
            # 可以保存失败案例到文件

        logger.info(f"放射组学特征提取完成: {len(radiomics_data)} 成功, {len(failed_cases)} 失败")
        return result_df

    def _find_mask_path(self, pid: str, mask_dir: Optional[str]) -> Optional[str]:
        """查找掩码路径"""
        if not mask_dir:
            return None

        mask_dir_path = Path(mask_dir)
        possible_names = [
            f"{pid}_mask.nii.gz",
            f"{pid}.nii.gz",
            f"mask_{pid}.nii.gz"
        ]

        for name in possible_names:
            mask_path = mask_dir_path / name
            if mask_path.exists():
                return str(mask_path)

        return None

    def _extract_with_mask(self, image_path: str, mask_path: str) -> Dict[str, float]:
        """使用掩码提取特征"""
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        # 提取特征
        features = self.extractor.execute(image, mask)

        # 清理特征名和值
        cleaned_features = {}
        for feature_name, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # 清理特征名，移除前缀
                clean_name = feature_name.replace('original_', '')
                cleaned_features[f'rad__{clean_name}'] = float(value)

        return cleaned_features

    def _extract_without_mask(self, image_path: str) -> Dict[str, float]:
        """不使用掩码提取特征（整个影像区域）"""
        image = sitk.ReadImage(image_path)

        # 创建全掩码（包含整个影像）
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
        mask = sitk.BinaryThreshold(mask, 0, 1000)

        # 提取特征
        features = self.extractor.execute(image, mask)

        # 清理特征名和值
        cleaned_features = {}
        for feature_name, value in features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                clean_name = feature_name.replace('original_', '')
                cleaned_features[f'rad__{clean_name}'] = float(value)

        return cleaned_features

    def normalize_features(self, radiomics_df: pd.DataFrame,
                          center_ids: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        标准化放射组学特征（处理中心效应）

        Args:
            radiomics_df: 包含放射组学特征的数据框
            center_ids: 中心ID序列

        Returns:
            pd.DataFrame: 标准化后的数据框
        """
        from sklearn.preprocessing import RobustScaler
        from combat.pycombat import pycombat

        # 获取放射组学特征列
        rad_cols = [col for col in radiomics_df.columns if col.startswith('rad__')]

        if len(rad_cols) == 0:
            logger.warning("没有找到放射组学特征列")
            return radiomics_df.copy()

        df_normalized = radiomics_df.copy()

        if center_ids is not None and self.center_correction:
            logger.info("使用ComBat方法进行中心效应校正")

            try:
                # 使用ComBat进行校正
                rad_data = radiomics_df[rad_cols].values.T  # 转置为特征×样本

                # 执行ComBat校正
                corrected_data = pycombat(
                    data=rad_data,
                    batch=center_ids.astype(str).values,
                    modality=None
                )

                # 更新数据框
                df_normalized[rad_cols] = corrected_data.T

            except ImportError:
                logger.warning("未安装combat包，跳过中心效应校正")
            except Exception as e:
                logger.warning(f"ComBat校正失败: {e}，使用RobustScaler")
                # 回退到鲁棒标准化
                scaler = RobustScaler()
                df_normalized[rad_cols] = scaler.fit_transform(df_normalized[rad_cols])
        else:
            # 使用鲁棒标准化
            logger.info("使用RobustScaler进行特征标准化")
            scaler = RobustScaler()
            df_normalized[rad_cols] = scaler.fit_transform(df_normalized[rad_cols])

            # 保存scaler
            self.fitted_scaler = scaler

        return df_normalized

    def select_features(self, radiomics_df: pd.DataFrame,
                       outcome_series: pd.Series,
                       method: str = 'mrmr',
                       n_features: int = 50) -> List[str]:
        """
        特征选择

        Args:
            radiomics_df: 放射组学特征数据框
            outcome_series: 结局变量
            method: 选择方法 ('mrmr', 'univariate', 'pca')
            n_features: 选择的特征数量

        Returns:
            List[str]: 选择的特征名称列表
        """
        rad_cols = [col for col in radiomics_df.columns if col.startswith('rad__')]

        if len(rad_cols) == 0:
            return []

        if method == 'mrmr':
            selected_features = self._mrmr_selection(
                radiomics_df[rad_cols], outcome_series, n_features
            )
        elif method == 'univariate':
            selected_features = self._univariate_selection(
                radiomics_df[rad_cols], outcome_series, n_features
            )
        elif method == 'pca':
            selected_features = self._pca_selection(
                radiomics_df[rad_cols], n_features
            )
        else:
            raise ValueError(f"Unsupported selection method: {method}")

        logger.info(f"特征选择完成，选择了 {len(selected_features)} 个特征")
        return selected_features

    def _mrmr_selection(self, features_df: pd.DataFrame,
                       outcome_series: pd.Series,
                       n_features: int) -> List[str]:
        """最大相关最小冗余特征选择"""
        try:
            import pymrmr

            # 合并特征和结局变量
            combined_df = features_df.copy()
            combined_df['outcome'] = outcome_series

            # 使用mRMR选择
            selected = pymrmr.mRMR(combined_df, 'MIQ', n_features)

            return selected

        except ImportError:
            logger.warning("未安装pymrmr包，回退到单变量选择")
            return self._univariate_selection(features_df, outcome_series, n_features)

    def _univariate_selection(self, features_df: pd.DataFrame,
                             outcome_series: pd.Series,
                             n_features: int) -> List[str]:
        """单变量特征选择"""
        from sklearn.feature_selection import mutual_info_classif, f_classif

        # 计算互信息
        mi_scores = mutual_info_classif(
            features_df.fillna(features_df.mean()),
            outcome_series
        )

        # 排序并选择
        feature_scores = list(zip(features_df.columns, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        selected_features = [feature for feature, score in feature_scores[:n_features]]

        return selected_features

    def _pca_selection(self, features_df: pd.DataFrame, n_components: int) -> List[str]:
        """PCA特征选择"""
        from sklearn.decomposition import PCA

        # 数据预处理
        features_clean = features_df.fillna(features_df.mean())

        # PCA降维
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features_clean)

        # 创建PCA特征名
        pca_cols = [f'rad__pca_{i}' for i in range(n_components)]

        # 保存PCA模型
        self.fitted_pca = pca

        return pca_cols

    def save_extractor(self, save_path: str):
        """保存提取器配置"""
        save_data = {
            'config': self.config,
            'feature_classes': self.feature_classes
        }

        # 保存scaler如果存在
        if hasattr(self, 'fitted_scaler'):
            save_data['scaler'] = self.fitted_scaler

        # 保存PCA如果存在
        if hasattr(self, 'fitted_pca'):
            save_data['pca'] = self.fitted_pca

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"放射组学提取器已保存: {save_path}")

    def load_extractor(self, load_path: str):
        """加载提取器配置"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)

        # 恢复属性
        if 'scaler' in save_data:
            self.fitted_scaler = save_data['scaler']

        if 'pca' in save_data:
            self.fitted_pca = save_data['pca']

        logger.info(f"放射组学提取器已加载: {load_path}")