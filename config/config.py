"""
项目配置文件
包含所有环境相关的配置项，支持开发、生产和测试环境的配置
使用环境变量进行敏感信息配置，通过.env文件管理
"""

import os
from dotenv import load_dotenv
import torch
from datetime import timedelta
import logging

# 加载环境变量
load_dotenv()

class Config:
    """
    基础配置类，包含所有环境的通用配置
    子类可以通过继承来覆盖特定环境的配置
    """
    
    # 应用配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'logs/app.log'
    LOG_TO_CONSOLE = True
    LOG_TO_FILE = False
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///prod.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # 禁用SQLAlchemy事件系统
    
    # 模型存储配置
    MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH') or 'models'
    MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练配置
    MAX_TRAINING_THREADS = int(os.getenv('MAX_TRAINING_THREADS', 4))  # 最大并行训练线程数
    
    # 服务配置
    DATA_SERVICE = {
        'processor': 'app.core.processors.default_data_processor.DefaultDataProcessor'
    }
    
    MODEL_SERVICE = {
        'model_ops': 'app.core.processors.default_model_operations.DefaultModelOperations'
    }
    
    TRAINING_SERVICE = {
        'training_impl': 'app.core.processors.default_training_service.DefaultTrainingService'
    }
    
    # Flask配置
    FLASK_DEBUG = False
    
class DevelopmentConfig(Config):
    """
    开发环境配置
    """
    DEBUG = True  # 启用调试模式
    FLASK_DEBUG = True
    LOG_LEVEL = logging.DEBUG
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///dev.db'
    SQLALCHEMY_ECHO = True  # 打印SQL语句

class ProductionConfig(Config):
    """
    生产环境配置
    """
    DEBUG = False  # 禁用调试模式
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///prod.db'
    # 生产环境下使用更严格的安全设置
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SECURE = True
    REMEMBER_COOKIE_HTTPONLY = True

class TestingConfig(Config):
    """
    测试环境配置
    """
    TESTING = True  # 启用测试模式
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or \
        'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False  # 禁用CSRF保护，方便测试

# 配置映射字典，用于根据环境名称选择配置
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
