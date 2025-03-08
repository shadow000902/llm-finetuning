from setuptools import setup, find_packages
import os
import re

# 读取版本号
with open("app/__init__.py", "r", encoding="utf-8") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

# 读取README文件
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# 定义依赖项
install_requires = [
    "flask>=2.0.0",
    "flask-sqlalchemy>=2.5.0",
    "torch>=1.9.0",
    "numpy>=1.19.0",
    "pandas>=1.0.0",
    "transformers>=4.5.0",
    "psutil>=5.8.0",
    "pyjwt>=2.1.0",
    "python-dotenv>=0.19.0",  # 添加dotenv依赖
    # 添加其他依赖项
]

# 定义开发依赖项
dev_requires = [
    "pytest>=6.0.0",
    "pytest-cov>=2.10.0",
    "black>=20.8b1",
    "isort>=5.0.0",
    "flake8>=3.8.0",
    # 添加其他开发依赖项
]

setup(
    name="ml-model-service",
    version=version,
    author="作者名称",
    author_email="作者邮箱",
    description="机器学习模型服务应用",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/用户名/ml-model-service",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: Flask",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "ml-service=app.cli:main",
        ],
    },
)
