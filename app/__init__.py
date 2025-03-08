"""
应用初始化模块
负责创建和配置Flask应用实例
包含应用扩展、服务、蓝图的初始化逻辑
"""

__version__ = "0.1.0"  # 添加版本号

import os
import logging
from flask import Flask
from app.extensions import db  # 导出db对象
from app.utils.logging import setup_logging
from app.utils.config_loader import load_config, get_config

def create_app(env: str = None):
    """
    应用工厂函数，创建并配置Flask应用实例
    
    Args:
        env: 环境名称，默认从环境变量FLASK_ENV获取，如果未设置则为development
        
    Returns:
        Flask: 配置完成的Flask应用实例
    """
    # 确定环境
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    
    # 加载配置
    try:
        from app.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.load_config(env)
        
        # 设置日志配置
        setup_logging(
            log_level=config.get('logging', {}).get('level', 'INFO'),
            log_format=config.get('logging', {}).get('format'),
            log_file=config.get('logging', {}).get('file') if config.get('logging', {}).get('to_file') else None
        )
    except Exception as e:
        # 确保即使配置加载失败，也能设置基本日志
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.error(f"配置加载失败: {str(e)}", exc_info=True)
        config = {}
    
    logger = logging.getLogger(__name__)
    logger.info(f"创建应用实例，环境: {env}")
    
    # 创建Flask应用实例
    app = Flask(__name__)
    
    # 设置基本配置
    app.config['SECRET_KEY'] = config.get('app', {}).get('secret_key', 'hard-to-guess-string')
    app.config['DEBUG'] = config.get('app', {}).get('debug', False)
    
    # 设置数据库配置
    app.config['SQLALCHEMY_DATABASE_URI'] = config.get('database', {}).get('url', 'sqlite:///app.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.get('database', {}).get('track_modifications', False)
    
    # 设置JWT配置
    app.config['JWT_SECRET_KEY'] = config.get('security', {}).get('jwt_secret_key', 'jwt-secret-key')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = config.get('security', {}).get('jwt_access_token_expires', 3600)
    
    # 存储配置到应用实例
    app.config['APP_CONFIG'] = config
    
    # 从config.py加载配置
    from config.config import Config
    for key in dir(Config):
        if key.isupper():
            app.config[key] = getattr(Config, key)
    
    # 初始化扩展
    init_extensions(app)
    
    # 注册蓝图
    register_blueprints(app)
    
    # 创建数据库表并初始化核心服务
    with app.app_context():
        init_database(app)
        init_services(app)
    
    logger.info("应用实例创建完成")
    return app

def init_extensions(app):
    """
    初始化Flask扩展
    
    Args:
        app: Flask应用实例
    """
    from app.extensions import db
    db.init_app(app)  # 初始化数据库扩展
    
    # 初始化其他扩展
    # 如果需要使用Flask-JWT-Extended
    try:
        from flask_jwt_extended import JWTManager
        jwt = JWTManager(app)
    except ImportError:
        pass
    
    # 如果需要使用Flask-CORS
    if app.config['APP_CONFIG'].get('api', {}).get('enable_cors', False):
        try:
            from flask_cors import CORS
            allowed_origins = app.config['APP_CONFIG'].get('api', {}).get('allowed_origins', ["*"])
            CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
        except ImportError:
            pass
    
    # 如果需要使用Flask-Limiter
    if app.config['APP_CONFIG'].get('api', {}).get('enable_rate_limit', False):
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address
            
            limiter = Limiter(
                app=app,
                key_func=get_remote_address,
                default_limits=[app.config['APP_CONFIG'].get('api', {}).get('rate_limit_default', "100/minute")]
            )
            app.extensions['limiter'] = limiter
        except ImportError:
            pass

def register_blueprints(app):
    """
    注册蓝图
    
    Args:
        app: Flask应用实例
    """
    # 注册API蓝图
    from app.api import bp as api_bp
    api_prefix = app.config['APP_CONFIG'].get('api', {}).get('prefix', '/api/v1')
    app.register_blueprint(api_bp, url_prefix=api_prefix)

def init_database(app):
    """
    初始化数据库
    
    Args:
        app: Flask应用实例
    """
    db.create_all()  # 在应用上下文中创建所有数据库表

def init_services(app):
    """
    初始化核心服务
    
    Args:
        app: Flask应用实例
    """
    # 初始化核心服务
    from app.core.factories.service_factory import ServiceFactory
    app.data_service = ServiceFactory.create_data_service()  # 数据服务
    app.model_service = ServiceFactory.create_model_service()  # 模型服务
    app.training_service = ServiceFactory.create_training_service()  # 训练服务
