"""
项目配置文件
包含所有环境相关的配置项，支持开发、生产和测试环境的配置
使用环境变量进行敏感信息配置，通过.env文件管理
"""

import os
from dotenv import load_dotenv
import torch

# 加载环境变量
load_dotenv()

class Config:
    """
    基础配置类，包含所有环境的通用配置
    子类可以通过继承来覆盖特定环境的配置
    """
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/dbname')
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # 禁用SQLAlchemy事件系统
    
    # JWT认证配置
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default-secret-key')  # JWT签名密钥
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1小时有效期
    
    # 模型存储配置
    MODEL_STORAGE_PATH = os.getenv('MODEL_STORAGE_PATH', './models')  # 模型文件存储路径
    MODEL_BASE_PATH = "/path/to/your/model/base"
    MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练配置
    MAX_TRAINING_THREADS = int(os.getenv('MAX_TRAINING_THREADS', 4))  # 最大并行训练线程数
    
    # 服务配置
    DATA_SERVICE = {
        'processor': 'app.services.model_components.data_processing.DataProcessor',  # 数据处理器实现类
        'cache_enabled': True,  # 是否启用缓存
        'cache_ttl': 3600  # 缓存有效期（秒）
    }
    
    MODEL_SERVICE = {
        'model_ops': 'app.services.model_components.model_operations.ModelOperations',  # 模型操作实现类
        'max_parallel_inference': 4  # 最大并行推理数
    }
    
    TRAINING_SERVICE = {
        'training_impl': 'app.services.model_components.model_training.ModelTrainer',  # 训练实现类
        'max_checkpoints': 5,  # 最大检查点保存数量
        'checkpoint_interval': 3600  # 检查点保存间隔（秒）
    }
    
class DevelopmentConfig(Config):
    """
    开发环境配置
    """
    DEBUG = True  # 启用调试模式
    SQLALCHEMY_ECHO = True  # 打印SQL语句

class ProductionConfig(Config):
    """
    生产环境配置
    """
    DEBUG = False  # 禁用调试模式
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')  # 必须从环境变量获取密钥

class TestingConfig(Config):
    """
    测试环境配置
    """
    TESTING = True  # 启用测试模式
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'  # 使用内存数据库
