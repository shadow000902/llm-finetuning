"""
应用初始化模块
负责创建和配置Flask应用实例
包含应用扩展、服务、蓝图的初始化逻辑
"""

from flask import Flask
from config import DevelopmentConfig

def create_app(config_class=DevelopmentConfig):
    """
    应用工厂函数，创建并配置Flask应用实例
    
    Args:
        config_class: 配置类，默认为开发环境配置
        
    Returns:
        Flask: 配置完成的Flask应用实例
    """
    # 创建Flask应用实例
    app = Flask(__name__)
    # 加载配置
    app.config.from_object(config_class)

    # 初始化扩展
    from app.extensions import db
    db.init_app(app)  # 初始化数据库扩展

    # 初始化核心服务
    from app.core.factories.service_factory import ServiceFactory
    app.data_service = ServiceFactory.create_data_service()  # 数据服务
    app.model_service = ServiceFactory.create_model_service()  # 模型服务
    app.training_service = ServiceFactory.create_training_service()  # 训练服务

    # 注册API蓝图
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api/v1')  # API前缀为/api/v1

    # 创建数据库表
    with app.app_context():
        db.create_all()  # 在应用上下文中创建所有数据库表

    return app
