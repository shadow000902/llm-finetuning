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
    
    @staticmethod
    def create_data_service() -> Any:
        """
        创建数据服务实例
        
        该方法根据配置创建数据服务实例，处理流程如下：
        1. 从应用配置获取数据服务配置
        2. 动态加载数据处理器类
        3. 实例化数据处理器
        4. 创建并返回数据服务实例
        
        Returns:
            DataService: 数据服务实例，包含以下功能：
            - 数据加载
            - 数据预处理
            - 数据增强
            - 数据缓存
            
        Raises:
            ImportError: 当无法加载数据处理器类时抛出
            ValueError: 当配置无效时抛出
        """
        config = current_app.config['DATA_SERVICE']
        processor_class = ServiceFactory._import_class(config['processor'])
        return DataService(processor_class())

    @staticmethod
    def create_model_service() -> Any:
        """
        创建模型服务实例
        
        该方法根据配置创建模型服务实例，处理流程如下：
        1. 从应用配置获取模型服务配置
        2. 动态加载模型操作接口实现类
        3. 实例化模型操作接口
        4. 创建并返回模型服务实例
        
        Returns:
            ModelService: 模型服务实例，包含以下功能：
            - 模型训练
            - 模型评估
            - 模型推理
            - 模型保存/加载
            
        Raises:
            ImportError: 当无法加载模型操作类时抛出
            ValueError: 当配置无效时抛出
        """
        config = current_app.config['MODEL_SERVICE']
        model_ops_class = ServiceFactory._import_class(config['model_ops'])
        return ModelService(model_ops_class())

    @staticmethod
    def create_training_service() -> Any:
        """
        创建训练服务实例
        
        该方法根据配置创建训练服务实例，处理流程如下：
        1. 从应用配置获取训练服务配置
        2. 动态加载训练实现类
        3. 实例化训练实现
        4. 创建并返回训练服务实例
        
        Returns:
            TrainingService: 训练服务实例，包含以下功能：
            - 训练过程管理
            - 资源监控
            - 早停机制
            - 训练报告生成
            
        Raises:
            ImportError: 当无法加载训练实现类时抛出
            ValueError: 当配置无效时抛出
        """
        config = current_app.config['TRAINING_SERVICE']
        training_impl_class = ServiceFactory._import_class(config['training_impl'])
        return TrainingService(training_impl_class())

    @staticmethod
    def _import_class(class_path: str) -> Any:
        """
        动态导入类
        
        该方法根据完整类路径动态加载Python类，处理流程如下：
        1. 解析模块路径和类名
        2. 导入目标模块
        3. 获取模块中的类
        4. 返回类引用
        
        Args:
            class_path (str): 类的完整路径，格式为'module.path.ClassName'
            
        Returns:
            Any: 导入的类引用
            
        Raises:
            ImportError: 当模块或类不存在时抛出
            AttributeError: 当模块中不存在指定类时抛出
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
