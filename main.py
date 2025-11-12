#!/usr/bin/env python3
"""
SaNa项目主入口脚本
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from etl.extract import ExcelExtractor
from etl.transform import DataTransformer
from etl.load import DataLoader
from features.build_t0 import T0FeatureBuilder
from features.build_img_embed import ImageEmbeddingBuilder
from models.train import ModelTrainer
from eval.cv_loco import CVLOCOEvaluator
from eval.reporting import ResultReporter

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sana.log')
        ]
    )

def run_etl(config_path: str):
    """运行ETL流程"""
    print("🔄 开始ETL流程...")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. 数据提取
    extractor = ExcelExtractor(config)
    all_data = extractor.extract_all_sheets()

    # 2. 数据转换
    transformer = DataTransformer(config)
    df_transformed = transformer.transform_all_centers(all_data)

    # 3. 数据加载
    load_config = config.copy()
    load_config['output_dir'] = 'data/artifacts'
    loader = DataLoader(load_config)
    tables = loader.load_data_contracts(df_transformed)

    # 4. 保存结果
    saved_paths = loader.save_tables(tables)

    print("✅ ETL流程完成!")
    return tables

def run_features(config_path: str):
    """运行特征工程"""
    print("🔧 开始特征工程...")

    # 加载ETL结果
    artifacts_dir = Path('data/artifacts')
    latest_artifact = max(artifacts_dir.iterdir()) if artifacts_dir.exists() else None

    if not latest_artifact:
        print("❌ 未找到ETL结果，请先运行ETL流程")
        return None

    # 加载数据表
    patients_df = pd.read_parquet(latest_artifact / 'patients.parquet')
    surgery_cpb_df = pd.read_parquet(latest_artifact / 'surgery_cpb.parquet')
    labs_long_df = pd.read_parquet(latest_artifact / 'labs_long.parquet')
    imaging_meta_df = pd.read_parquet(latest_artifact / 'imaging_meta.parquet')
    outcomes_df = pd.read_parquet(latest_artifact / 'outcomes.parquet')

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 构建T0特征
    t0_builder = T0FeatureBuilder(config.get('features', {}))
    t0_features = t0_builder.build_t0_features(
        patients_df, surgery_cpb_df, labs_long_df, imaging_meta_df
    )

    # 构建影像嵌入（如果有影像数据）
    image_features = None
    img_config = config.get('image_features', {})
    if img_config:
        img_builder = ImageEmbeddingBuilder(img_config)
        image_features = img_builder.build_embeddings(imaging_meta_df)

    print("✅ 特征工程完成!")
    return t0_features, image_features, outcomes_df

def run_training(config_path: str, view: str = "Primary_set_img"):
    """运行模型训练"""
    print("🤖 开始模型训练...")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 准备数据
    feature_data = run_features(config_path)
    if not feature_data:
        print("❌ 特征工程失败，无法训练模型")
        return

    t0_features, image_features, outcomes_df = feature_data

    # 合并数据
    surgery_cpb_df = pd.read_parquet(Path('data/artifacts') / max(Path('data/artifacts').iterdir()) / 'surgery_cpb.parquet')
    patients_df = pd.read_parquet(Path('data/artifacts') / max(Path('data/artifacts').iterdir()) / 'patients.parquet')

    # 训练模型
    trainer = ModelTrainer(config)
    trainer.setup_models()

    # 准备训练数据
    prepared_data = trainer.prepare_data(
        patients_df, surgery_cpb_df,
        pd.read_parquet(Path('data/artifacts') / max(Path('data/artifacts').iterdir()) / 'labs_long.parquet'),
        pd.read_parquet(Path('data/artifacts') / max(Path('data/artifacts').iterdir()) / 'imaging_meta.parquet'),
        outcomes_df
    )

    # 训练所有模型
    training_results = trainer.train_all_models(prepared_data)

    # 保存结果
    results_dir = trainer.save_results()
    model_cards = trainer.generate_model_cards()

    print("✅ 模型训练完成!")
    return training_results, results_dir

def run_evaluation(config_path: str, results_dir: str):
    """运行模型评估"""
    print("📊 开始模型评估...")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建评估器
    evaluator = CVLOCOEvaluator(config.get('evaluation', {}))

    # 生成报告
    reporter = ResultReporter()

    print("✅ 模型评估完成!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="SaNa - A型主动脉夹层多中心AI预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py etl --config configs/etl.yml
  python main.py train --config configs/train_primary.yml
  python main.py evaluate --config configs/train_primary.yml --results results/
        """
    )

    parser.add_argument(
        'command',
        choices=['etl', 'features', 'train', 'evaluate', 'all'],
        help='要执行的命令'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_primary.yml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--view',
        type=str,
        default='Primary_set_img',
        help='数据视图名称'
    )

    parser.add_argument(
        '--results',
        type=str,
        default='results/',
        help='结果目录'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    print("🚀 SaNa - A型主动脉夹层多中心AI预测系统")
    print("=" * 50)

    try:
        if args.command == 'etl':
            run_etl('configs/etl.yml')

        elif args.command == 'features':
            run_features(args.config)

        elif args.command == 'train':
            run_training(args.config, args.view)

        elif args.command == 'evaluate':
            run_evaluation(args.config, args.results)

        elif args.command == 'all':
            # 完整流程
            print("运行完整流程: ETL -> 特征工程 -> 模型训练")

            # 1. ETL
            tables = run_etl('configs/etl.yml')

            # 2. 特征工程
            feature_data = run_features(args.config)

            # 3. 模型训练
            if feature_data:
                training_results, results_dir = run_training(args.config, args.view)

                # 4. 模型评估
                if results_dir:
                    run_evaluation(args.config, results_dir)

        print("🎉 所有任务完成!")

    except Exception as e:
        logging.error(f"❌ 执行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()