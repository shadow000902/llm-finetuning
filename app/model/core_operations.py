from typing import Dict, Any
from app.core.services.model_service import ModelService
from app.core.services.training_service import TrainingService
from app.core.services.data_service import DataService

class ModelCoreOperations:
    """模型核心操作类，负责协调模型服务、训练服务和数据服务"""
    
    def __init__(self, model_service: ModelService, training_service: TrainingService, data_service: DataService):
        """初始化核心操作
        
        Args:
            model_service: 模型服务实例
            training_service: 训练服务实例
            data_service: 数据服务实例
        """
        self._model_service = model_service
        self._training_service = training_service
        self._data_service = data_service

    def initialize_model(self, model_config: Dict[str, Any]) -> None:
        """初始化模型
        
        Args:
            model_config: 模型配置字典
            
        Raises:
            ValueError: 当配置无效时抛出
        """
        self._model_service.initialize(model_config)

    def train_model(self, training_data: Any, training_config: Dict[str, Any], data_processor) -> Dict[str, Any]:
        """训练模型
        
        Args:
            training_data: 训练数据
            training_config: 训练配置
            data_processor: 数据处理器实例
            
        Returns:
            包含训练结果的字典
            
        Raises:
            RuntimeError: 当训练失败时抛出
        """
        # 数据预处理
        processed_data = data_processor.preprocess_data(training_data)
        
        # 配置训练
        self._training_service.configure(training_config)
        
        # 执行训练
        return self._training_service.train(processed_data)

    def predict(self, input_data: Any, data_processor) -> Any:
        """使用模型进行预测
        
        Args:
            input_data: 输入数据
            data_processor: 数据处理器实例
            
        Returns:
            预测结果
            
        Raises:
            ValueError: 当输入数据无效时抛出
            RuntimeError: 当预测失败时抛出
        """
        # 数据预处理
        processed_data = data_processor.preprocess_data(input_data)
        
        # 执行预测
        return self._model_service.predict(processed_data)
