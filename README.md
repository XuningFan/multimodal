# SaNa - Spec-Kit for Aortic Dissection Analysis

A型主动脉夹层多中心队列研究：基于规格驱动的AI模型预测系统

## 项目概述

**SaNa**是一个专门为Type A Aortic Dissection (AAD)多中心研究设计的AI预测系统。本项目采用规格驱动开发方法，确保从数据到论文的端到端流程可复现、可验证、可上线。

### 核心功能

- **多中心数据整合**: 整合4个医疗中心的Excel数据，标准化为统一数据契约
- **多模态融合**: 结合临床数据和术前CT影像进行预测
- **三级融合策略**: 早期融合、中期融合、晚期融合的系统比较
- **严格验证**: 5折交叉验证 + LOCO (Leave-One-Center-Out)验证
- **临床可解释性**: 提供特征重要性分析和模型卡片

## 主要预测任务

### 主问题 (Primary)
- **30天死亡预测**: outcome__death30
- **主要不良事件 (MAE)**: outcome__mae

### 次问题 (Secondary)
- 急性肾损伤 (AKI): outcome__aki
- 呼吸衰竭: outcome__resp_fail
- 肝功能损伤: outcome__liver_injury
- 大出血: outcome__major_bleed
- 脑卒中: outcome__stroke

## 技术架构

```
数据流程: Excel(4中心) → ETL → 数据契约 → 特征工程 → 多模态融合 → 模型训练 → 评估 → 报告
```

### 核心模块

1. **ETL模块** (`etl/`): 数据提取、转换、加载
2. **特征工程** (`features/`): T0特征构建、影像嵌入、放射组学
3. **模型开发** (`models/`): 临床基线、三种融合策略
4. **评估模块** (`eval/`): CV/LOCO验证、统计检验、可视化
5. **配置管理** (`configs/`): YAML配置文件

## 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 1.9
pandas, numpy, scikit-learn
lightgbm, xgboost
SimpleITK, PyRadiomics
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

1. 将多中心Excel数据放入 `data/raw/AAD_multi_center.xlsx`
2. 确保数据包含以下工作表: CenterA, CenterB, CenterC, CenterD

### 运行ETL流程

```bash
# 1. 运行ETL
python -m etl.extract --config configs/etl.yml
python -m etl.transform --config configs/etl.yml
python -m etl.load --config configs/etl.yml

# 2. 构建特征
python -m features.build_t0 --config configs/features.yml
python -m features.build_img_embed --subset Imaging_subset

# 3. 训练主问题模型
python -m models.train --config configs/train_primary.yml --view Primary_set_img

# 4. 评估模型
python -m eval.cv_loco --runs last --report reports/results_primary.md
```

## 配置说明

### 主要配置文件

- `configs/etl.yml`: ETL流程配置
- `configs/train_primary.yml`: 主问题训练配置
- `configs/train_secondary_aki.yml`: AKI次问题配置
- `configs/tasks.yml`: 任务定义配置

### 关键配置项

```yaml
# ETL配置
source_excel: data/raw/AAD_multi_center.xlsx
blacklist_features:
  - lab__*_d1  # 防术后特征泄漏
  - lab__*_d2

# 模型配置
models:
  clinical_only:
    type: lightgbm
  fusion_early:
    type: early
    img_backbone: resnet18
```

## 数据契约

### 标准化命名规范

`<domain>__<field>__[timepoint]`

示例:
- `lab__Cr__preop`: 术前肌酐
- `cpb__time_min`: CPB时间
- `demo__age_yr`: 患者年龄

### 输出表结构

1. **patients**: 患者基础信息
2. **surgery_cpb**: 手术和CPB信息
3. **labs_long**: 化验检查长表
4. **imaging_meta**: 影像元数据
5. **outcomes**: 结局指标
6. **meta_center**: 中心元数据

## 模型架构

### 1. 临床基线模型
- LightGBM/XGBoost/Logistic回归
- 仅使用临床数据

### 2. 早期融合 (Early Fusion)
- 特征级别连接: ImageEmbedding + TabularMLP → 分类头
- 参数量: 20-30M

### 3. 中期融合 (Intermediate Fusion)
- 跨注意力机制: ImageEnc + TabMLP → CrossAttention → 分类头
- 支持单向/双向交互

### 4. 晚期融合 (Late Fusion)
- 决策级别: 独立训练 → Stacking/加权平均
- 作为稳定基线

## 验证策略

### 1. 内部验证
- 5折分层交叉验证
- Bootstrap置信区间 (1000次)

### 2. 外部验证
- LOCO (Leave-One-Center-Out)
- 跨中心泛化性评估

### 3. 统计检验
- DeLong检验 (AUC差异)
- NRI/IDI (重新分类分析)
- 多重比较校正

## 评估指标

- **区分度**: AUROC, AUPRC
- **校准度**: Brier分数, 校准曲线
- **临床效用**: 决策曲线分析
- **可解释性**: SHAP值, Grad-CAM

## 项目结构

```
SaNa/
├── configs/          # 配置文件
├── data/            # 数据目录
│   ├── raw/         # 原始数据
│   └── artifacts/   # ETL输出
├── etl/             # ETL模块
├── features/        # 特征工程
├── models/          # 模型开发
├── eval/            # 评估模块
├── tests/           # 测试用例
├── reports/         # 结果报告
├── info/            # 项目文档
└── CLAUDE.md        # AI助手指导
```

## 开发规范

### 规格驱动开发
- 严格遵循 `info/spec_kit.txt` 中的技术规格
- 数据契约确保跨中心一致性
- 防泄漏守卫保证模型有效性

### 代码质量
- 单元测试覆盖关键功能
- 集成测试验证端到端流程
- 合同测试确保数据结构一致性

### 可复现性
- 固定随机种子
- 版本化配置和产物
- 完整的实验记录

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 开发工作流

```bash
# 运行测试
pytest tests/

# 代码格式化
black etl/ features/ models/ eval/

# 类型检查
mypy etl/
```

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 引用

如果您使用了本项目的代码或数据，请引用:

```bibtex
@software{sana_aortic_dissection,
  title={SaNa: Spec-Kit for Aortic Dissection Analysis},
  author={[作者姓名]},
  year={2025},
  url={https://github.com/[username]/SaNa}
}
```

## 联系方式

- 项目维护者: [维护者姓名]
- 邮箱: [email]
- 项目主页: [GitHub链接]

## 致谢

感谢所有参与多中心数据收集的医疗机构和研究人员。

---

**注意**: 本项目仅用于研究目的，不应直接用于临床决策。临床使用前需要额外的验证和监管批准。