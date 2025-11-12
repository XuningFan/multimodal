#!/bin/bash

# SaNa项目环境激活脚本
# 使用方法: source activate_env.sh

echo "🔧 激活SaNa项目Python环境..."

# 获取conda路径
CONDA_PATH="/home/fanxn/miniconda3"
ENV_NAME="multimodal"

# 检查conda是否存在
if [ ! -f "$CONDA_PATH/bin/conda" ]; then
    echo "❌ 错误: 找不到conda安装路径: $CONDA_PATH"
    return 1
fi

# 检查环境是否存在
if [ ! -d "$CONDA_PATH/envs/$ENV_NAME" ]; then
    echo "❌ 错误: conda环境 $ENV_NAME 不存在"
    echo "请先运行: conda create -n $ENV_NAME python=3.9"
    return 1
fi

# 激活环境
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "✅ 环境 $ENV_NAME 已激活"
echo "Python版本: $(python --version)"
echo "Python路径: $(which python)"

# 显示已安装的主要包版本
echo ""
echo "📦 已安装的主要包版本:"
python -c "
import pkg_resources
packages = ['numpy', 'pandas', 'torch', 'scikit-learn', 'matplotlib', 'lightgbm', 'xgboost']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'  ✓ {pkg}: {version}')
    except:
        print(f'  ✗ {pkg}: 未安装')
"

echo ""
echo "🚀 SaNa项目环境已准备就绪!"
echo "现在可以运行项目，例如:"
echo "  python main.py --help"
echo "  jupyter notebook  # 如果需要Jupyter环境"