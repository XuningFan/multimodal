from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sana-aortic-dissection",
    version="1.0.0",
    author="SaNa Research Team",
    author_email="research@sana-project.org",
    description="Spec-Kit for Aortic Dissection Analysis - Multi-center AI prediction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sana-project/sana-aortic-dissection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "tracking": [
            "wandb>=0.12.0",
            "mlflow>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sana-etl=etl.cli:main",
            "sana-train=models.cli:main",
            "sana-eval=eval.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "configs": ["*.yml"],
        "info": ["*.txt", "*.pdf"],
    },
    keywords="medical-ai, aortic-dissection, machine-learning, multimodal-fusion",
    project_urls={
        "Bug Reports": "https://github.com/sana-project/sana-aortic-dissection/issues",
        "Source": "https://github.com/sana-project/sana-aortic-dissection",
        "Documentation": "https://sana-aortic-dissection.readthedocs.io/",
    },
)