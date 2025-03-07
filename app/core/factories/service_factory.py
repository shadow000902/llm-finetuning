"""
服务工厂模块
采用工厂模式创建和管理各种服务实例
支持动态加载和配置服务实现类
"""

from importlib import import_module
from typing import Any
from flask import current_app
from app.core.services.model_service import ModelService

class ServiceFactory:
    """
    服务工厂类
    负责创建数据服务、模型服务和训练服务实例
    通过配置动态加载具体实现类
    """
    
    @staticmethod
    def create_data_service() -> Any:
        """
        创建数据服务实例
        
        Returns:
            DataService: 数据服务实例
        """
        config = current_app.config['DATA_SERVICE']
        processor_class = ServiceFactory._import_class(config['processor'])
        return DataService(processor_class())

    @staticmethod
    def create_model_service() -> Any:
        """
        创建模型服务实例
        
        Returns:
            ModelService: 模型服务实例
        """
        config = current_app.config['MODEL_SERVICE']
        model_ops_class = ServiceFactory._import_class(config['model_ops'])
        return ModelService(model_ops_class())

    @staticmethod
    def create_training_service() -> Any:
        """
        创建训练服务实例
        
        Returns:
            TrainingService: 训练服务实例
        """
        config = current_app.config['TRAINING_SERVICE']
        training_impl_class = ServiceFactory._import_class(config['training_impl'])
        return TrainingService(training_impl_class())

    @staticmethod
    def _import_class(class_path: str) -> Any:
        """
        动态导入类
        
        Args:
            class_path (str): 类的完整路径，格式为'module.path.ClassName'
            
        Returns:
            Any: 导入的类
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
