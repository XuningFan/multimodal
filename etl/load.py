"""
数据加载模块：将转换后的数据保存为标准化的表格格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器，负责保存标准化数据表"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化加载器

        Args:
            config: 包含输出路径等配置
        """
        self.output_dir = Path(config['output_dir'])
        self.version = datetime.now().strftime("%Y%m%d_%H%M")
        self.versioned_dir = self.output_dir / self.version
        self.versioned_dir.mkdir(parents=True, exist_ok=True)

    def load_data_contracts(self, df_transformed: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        将转换后的数据分解为标准化的数据契约表

        Args:
            df_transformed: 转换后的完整数据框

        Returns:
            Dict[str, pd.DataFrame]: 数据契约表字典
        """
        logger.info("开始生成数据契约表")

        # 1. 患者基础信息表
        patients_df = self._create_patients_table(df_transformed)

        # 2. 手术和CPB信息表
        surgery_cpb_df = self._create_surgery_cpb_table(df_transformed)

        # 3. 化验检查长格式表
        labs_long_df = self._create_labs_long_table(df_transformed)

        # 4. 影像元数据表
        imaging_meta_df = self._create_imaging_meta_table(df_transformed)

        # 5. 结局指标表
        outcomes_df = self._create_outcomes_table(df_transformed)

        # 6. 中心元数据表
        meta_center_df = self._create_meta_center_table(df_transformed)

        tables = {
            'patients': patients_df,
            'surgery_cpb': surgery_cpb_df,
            'labs_long': labs_long_df,
            'imaging_meta': imaging_meta_df,
            'outcomes': outcomes_df,
            'meta_center': meta_center_df
        }

        logger.info(f"生成 {len(tables)} 张数据契约表")
        return tables

    def _create_patients_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建患者基础信息表

        包含：pid, center_id, 人口学信息, 既往史
        """
        # 基础列
        base_cols = ['pid', 'center_id']

        # 人口学信息
        demo_cols = [col for col in df.columns if col.startswith('demo__')]

        # 既往史
        mhx_cols = [col for col in df.columns if col.startswith('mhx__')]

        # 入院信息
        admission_cols = [col for col in df.columns if col.startswith('admission__')]

        patients_cols = base_cols + demo_cols + mhx_cols + admission_cols
        patients_cols = [col for col in patients_cols if col in df.columns]

        patients_df = df[patients_cols].drop_duplicates(subset=['pid'])

        logger.info(f"患者表: {len(patients_df)} 患者的基础信息")
        return patients_df

    def _create_surgery_cpb_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建手术和CPB信息表

        包含：pid, 手术时间信息, CPB参数, 输血信息
        """
        base_cols = ['pid']

        # 手术相关
        surg_cols = [col for col in df.columns if col.startswith('surg__')]

        # CPB相关
        cpb_cols = [col for col in df.columns if col.startswith('cpb__')]

        # 灌注相关
        perf_cols = [col for col in df.columns if col.startswith('perf__')]

        # 输血相关
        transfusion_cols = [col for col in df.columns if col.startswith('transfusion__')]

        surgery_cpb_cols = base_cols + surg_cols + cpb_cols + perf_cols + transfusion_cols
        surgery_cpb_cols = [col for col in surgery_cpb_cols if col in df.columns]

        surgery_cpb_df = df[surgery_cpb_cols].drop_duplicates(subset=['pid'])

        logger.info(f"手术CPB表: {len(surgery_cpb_df)} 患者的手术信息")
        return surgery_cpb_df

    def _create_labs_long_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建化验检查长格式表

        格式：pid, timepoint, analyte, value, unit
        """
        labs_data = []

        # 识别化验相关列
        lab_cols = [col for col in df.columns if col.startswith('lab__')]

        for col in lab_cols:
            # 解析列名
            parts = col.split('__')
            if len(parts) >= 3:
                _, analyte, timepoint = parts[0], parts[1], parts[2]

                # 提取该化验项的所有值
                for idx, row in df.iterrows():
                    value = row[col]
                    if pd.notna(value):
                        labs_data.append({
                            'pid': row['pid'],
                            'timepoint': timepoint,
                            'analyte': analyte,
                            'value': value,
                            'unit': self._get_lab_unit(analyte)
                        })

        labs_long_df = pd.DataFrame(labs_data)

        logger.info(f"化验长表: {len(labs_long_df)} 条记录")
        return labs_long_df

    def _get_lab_unit(self, analyte: str) -> str:
        """获取化验项目的标准单位"""
        unit_mapping = {
            'Cr': 'μmol/L',
            'ALT': 'U/L',
            'AST': 'U/L',
            'TBil': 'μmol/L',
            'WBC': '×10^9/L',
            'Hb': 'g/L',
            'PLT': '×10^9/L',
            'Lactate': 'mmol/L'
        }
        return unit_mapping.get(analyte, 'unknown')

    def _create_imaging_meta_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建影像元数据表

        包含：pid, 影像可用性, 影像路径
        """
        base_cols = ['pid']

        # 影像相关列
        img_cols = [col for col in df.columns if col.startswith('img__')]

        imaging_cols = base_cols + img_cols
        imaging_cols = [col for col in imaging_cols if col in df.columns]

        if imaging_cols:
            imaging_meta_df = df[imaging_cols].drop_duplicates(subset=['pid'])
        else:
            # 如果没有影像列，创建基础表
            imaging_meta_df = df[base_cols].drop_duplicates(subset=['pid'])
            # 添加影像可用性标记（默认为0）
            imaging_meta_df['img__cta_preop_available'] = 0

        logger.info(f"影像表: {len(imaging_meta_df)} 患者的影像信息")
        return imaging_meta_df

    def _create_outcomes_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建结局指标表

        包含：pid, 各种死亡和并发症结局
        """
        base_cols = ['pid']

        # 结局相关列
        outcome_cols = [col for col in df.columns if col.startswith('outcome__')]

        outcomes_cols = base_cols + outcome_cols
        outcomes_cols = [col for col in outcomes_cols if col in df.columns]

        outcomes_df = df[outcomes_cols].drop_duplicates(subset=['pid'])

        logger.info(f"结局表: {len(outcomes_df)} 患者的结局信息")
        return outcomes_df

    def _create_meta_center_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建中心元数据表

        包含：中心基础信息和设备信息
        """
        # 获取唯一中心
        centers = df['center_id'].unique()

        center_data = []
        for center_id in centers:
            center_info = {
                'center_id': center_id,
                'patient_count': (df['center_id'] == center_id).sum(),
                # 这里可以添加更多中心级别的元数据
                'scanner_vendor': 'unknown',
                'scanner_model': 'unknown'
            }
            center_data.append(center_info)

        meta_center_df = pd.DataFrame(center_data)

        logger.info(f"中心元表: {len(meta_center_df)} 个中心的信息")
        return meta_center_df

    def save_tables(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        保存所有数据表

        Args:
            tables: 数据表字典

        Returns:
            Dict[str, str]: 保存的文件路径
        """
        saved_paths = {}

        for table_name, df in tables.items:
            # 保存为Parquet格式（高效，支持数据类型）
            parquet_path = self.versioned_dir / f"{table_name}.parquet"
            df.to_parquet(parquet_path, index=False)

            # 同时保存为CSV格式（便于查看）
            csv_path = self.versioned_dir / f"{table_name}.csv"
            df.to_csv(csv_path, index=False)

            saved_paths[table_name] = {
                'parquet': str(parquet_path),
                'csv': str(csv_path)
            }

            logger.info(f"保存表 {table_name}: {len(df)} 行")

        return saved_paths

    def generate_etl_report(self, tables: Dict[str, pd.DataFrame],
                          validation_results: Dict[str, Any]) -> str:
        """
        生成ETL报告

        Args:
            tables: 数据表字典
            validation_results: 验证结果

        Returns:
            str: 报告文件路径
        """
        report_path = self.versioned_dir / "etl_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# ETL报告\n\n")
            f.write(f"生成时间: {datetime.now()}\n\n")
            f.write(f"版本号: {self.version}\n\n")

            # 数据概览
            f.write("## 数据概览\n\n")
            for table_name, df in tables.items():
                f.write(f"- {table_name}: {len(df)} 行, {len(df.columns)} 列\n")

            # 验证结果
            f.write("\n## 数据质量验证\n\n")
            f.write(f"- 总记录数: {validation_results.get('total_rows', 'N/A')}\n")
            f.write(f"- 总列数: {validation_results.get('total_columns', 'N/A')}\n")
            f.write(f"- 唯一患者数: {validation_results.get('pid_unique_count', 'N/A')}\n")

            # 值域违规
            violations = validation_results.get('value_range_violations', [])
            if violations:
                f.write("\n### 值域违规检查\n\n")
                for violation in violations:
                    f.write(f"- {violation['column']}: {violation['violations']} 个违规值 "
                           f"(范围: {violation['range']})\n")

        logger.info(f"ETL报告已生成: {report_path}")
        return str(report_path)

    def save_metadata(self, config: Dict[str, Any],
                     validation_results: Dict[str, Any]) -> str:
        """
        保存元数据信息

        Args:
            config: 配置信息
            validation_results: 验证结果

        Returns:
            str: 元数据文件路径
        """
        metadata = {
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'validation_results': validation_results
        }

        metadata_path = self.versioned_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"元数据已保存: {metadata_path}")
        return str(metadata_path)