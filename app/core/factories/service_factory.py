"""
服务工厂模块
采用工厂模式创建和管理各种服务实例
支持动态加载和配置服务实现类
"""

from importlib import import_module
from typing import Any, Dict, Optional
import logging

from flask import current_app

from app.core.services.data_service import DataService
from app.core.services.model_service import ModelService
from app.core.services.training_service import TrainingService

logger = logging.getLogger(__name__)

class ServiceFactory:
    """
    服务工厂类
    
    采用工厂模式创建和管理各种服务实例，主要功能包括：
    - 根据配置动态创建服务实例
    - 管理服务生命周期
    - 提供统一的创建接口
    
    支持创建的服务类型：
    - 数据服务：负责数据预处理和加载
    - 模型服务：实现模型训练、评估和推理
    - 训练服务：管理训练过程和资源
    
    设计模式：
    - 工厂模式：封装对象创建逻辑
    - 依赖注入：通过配置动态加载实现类
    - 单例模式：确保服务实例的唯一性
    
    典型使用场景：
    1. 应用启动时初始化服务
    2. 根据环境配置创建不同实现
    3. 单元测试时注入mock服务
    """
    
    _instances = {}  # 服务实例缓存，实现单例模式
    
    @classmethod
    def create_data_service(cls) -> Any:
        """创建数据服务实例"""
        return cls._get_or_create_service('data_service', DataService)

    @classmethod
    def create_model_service(cls) -> Any:
        """创建模型服务实例"""
        return cls._get_or_create_service('model_service', ModelService)

    @classmethod
    def create_training_service(cls) -> Any:
        """创建训练服务实例"""
        return cls._get_or_create_service('training_service', TrainingService)

    @classmethod
    def _get_or_create_service(cls, service_type: str, service_class: Any) -> Any:
        """获取或创建服务实例，实现单例模式
        
        Args:
            service_type: 服务类型标识
            service_class: 服务类
            
        Returns:
            服务实例
        """
        # 检查缓存中是否已存在实例
        if service_type in cls._instances:
            return cls._instances[service_type]
            
        try:
            # 统一配置键名格式
            config_key = service_type.upper()
            # 尝试从应用配置中获取服务配置
            config = current_app.config.get(config_key, {})
            
            # 如果没有找到配置，使用默认实现
            if not config:
                logger.warning(f"未找到{config_key}的配置，将使用默认实现")
                instance = cls._create_default_service(service_class)
                cls._instances[service_type] = instance
                return instance
                
            impl_class_path = None
            for key in ['impl', 'implementation', 'processor', 'model_ops', 'training_impl']:
                if key in config:
                    impl_class_path = config[key]
                    break
                    
            if not impl_class_path:
                logger.warning(f"No implementation class specified in {config_key}, using defaults")
                instance = cls._create_default_service(service_class)
                cls._instances[service_type] = instance
                return instance
                
            try:
                impl_class = cls._import_class(impl_class_path)
                instance = service_class(impl_class())
                cls._instances[service_type] = instance
                return instance
            except Exception as e:
                logger.error(f"导入实现类失败: {impl_class_path}, 错误: {str(e)}")
                logger.info(f"将使用默认实现替代")
                instance = cls._create_default_service(service_class)
                cls._instances[service_type] = instance
                return instance
            
        except Exception as e:
            logger.error(f"创建服务失败 {service_type}: {str(e)}", exc_info=True)
            raise ValueError(f"创建服务失败 {service_type}: {str(e)}")

    @staticmethod
    def _import_class(class_path: str) -> Any:
        """动态导入类
        
        Args:
            class_path: 类的完整路径，格式为'module.path.ClassName'
            
        Returns:
            导入的类引用
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import {class_path}: {str(e)}", exc_info=True)
            raise ImportError(f"Failed to import {class_path}: {str(e)}")
            
    @classmethod
    def _create_default_service(cls, service_class: Any) -> Any:
        """创建默认服务实例
        
        Args:
            service_class: 服务类
            
        Returns:
            默认服务实例
        """
        if service_class.__name__ == 'DataService':
            # 为DataService创建一个默认的处理器
            from app.core.processors.default_data_processor import DefaultDataProcessor
            return service_class(DefaultDataProcessor())
        elif service_class.__name__ == 'ModelService':
            # 为ModelService创建一个默认的模型操作
            from app.core.processors.default_model_operations import DefaultModelOperations
            return service_class(DefaultModelOperations())
        elif service_class.__name__ == 'TrainingService':
            # 为TrainingService创建一个默认的训练服务
            from app.core.processors.default_training_service import DefaultTrainingService
            return service_class(DefaultTrainingService())
        else:
            # 对于其他服务类，尝试无参数初始化
            return service_class()
