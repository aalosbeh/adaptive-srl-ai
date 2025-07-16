#!/usr/bin/env python3
"""
Adaptive Multi-Modal AI Framework for Self-Regulated Learning
Setup script for package installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-srl-ai",
    version="1.0.0",
    author="Manus AI Research Team",
    author_email="research@manus.ai",
    description="Adaptive Multi-Modal AI Framework for Personalized Self-Regulated Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/manus-ai/adaptive-srl-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements() if os.path.exists("requirements.txt") else [
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "networkx>=2.8.0",
        "torch-geometric>=2.1.0",
        "gymnasium>=0.26.0",
        "stable-baselines3>=1.6.0",
        "tensorboard>=2.9.0",
        "wandb>=0.13.0",
        "opencv-python>=4.6.0",
        "pillow>=9.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "jsonschema>=4.0.0",
        "cryptography>=37.0.0",
        "differential-privacy>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.19.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptive-srl-train=adaptive_srl_ai.cli:train_model",
            "adaptive-srl-evaluate=adaptive_srl_ai.cli:evaluate_model",
            "adaptive-srl-federated=adaptive_srl_ai.cli:run_federated_learning",
        ],
    },
    include_package_data=True,
    package_data={
        "adaptive_srl_ai": [
            "configs/*.yaml",
            "data/sample_datasets/*.csv",
            "models/pretrained/*.pth",
        ],
    },
    zip_safe=False,
    keywords="artificial intelligence, education, self-regulated learning, federated learning, deep reinforcement learning, multi-modal AI",
    project_urls={
        "Bug Reports": "https://github.com/manus-ai/adaptive-srl-ai/issues",
        "Source": "https://github.com/manus-ai/adaptive-srl-ai",
        "Documentation": "https://adaptive-srl-ai.readthedocs.io/",
    },
)

