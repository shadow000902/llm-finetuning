from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np

class IModelOperations(ABC):
    """模型操作接口，定义模型训练、评估、预测等核心操作"""
    @abstractmethod
    def train(self,
             train_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
             val_data: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             early_stopping_patience: int = 5) -> Dict[str, any]:
        """训练模型"""
        pass

    @abstractmethod
    def evaluate(self,
                data: Union[torch.Tensor, Dict[str, torch.Tensor]],
                batch_size: int = 32) -> Dict[str, any]:
        """评估模型性能"""
        pass

    @abstractmethod
    def predict(self,
               inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
               batch_size: int = 32,
               return_probs: bool = False) -> np.ndarray:
        """模型推理"""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """保存模型状态"""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """加载模型状态"""
        pass

    @abstractmethod
    def generate_text(self,
                     prompt: str,
                     max_length: int = 50,
                     temperature: float = 0.7) -> str:
        """文本生成"""
        pass

class ITrainingService(ABC):
    """训练服务接口，定义模型训练相关操作"""
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """配置训练参数"""
        pass

    @abstractmethod
    def train(self,
             train_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
             val_data: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             early_stopping_patience: int = 5) -> Dict[str, Any]:
        """训练模型"""
        pass

class IModelService(ABC):
    """模型服务接口，定义模型操作的标准方法
    
    该接口定义了模型训练、评估、预测等核心操作，包括：
    - 模型训练
    - 模型评估
    - 模型推理
    - 模型保存与加载
    - 文本生成
    
    实现类需要提供具体的模型操作逻辑，适用于不同的模型架构
    """
    @abstractmethod
    def train(self,
             train_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
             val_data: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             early_stopping_patience: int = 5) -> Dict[str, any]:
        """训练模型
        
        Args:
            train_data: 训练数据，支持Tensor或字典格式
            val_data: 验证数据，可选
            epochs: 训练轮数，默认10
            batch_size: 批次大小，默认32
            learning_rate: 学习率，默认0.001
            early_stopping_patience: 早停耐心值，默认5
            
        Returns:
            包含训练结果的字典，通常包括：
            - train_loss: 训练损失
            - val_loss: 验证损失
            - metrics: 评估指标
            
        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            RuntimeError: 当训练过程中出现错误时抛出
            
        Example:
            >>> model = ConcreteModelService()
            >>> train_data = {"input_ids": torch.tensor(...), "labels": torch.tensor(...)}
            >>> results = model.train(train_data, epochs=5)
        """
        pass

    @abstractmethod
    def evaluate(self,
                data: Union[torch.Tensor, Dict[str, torch.Tensor]],
                batch_size: int = 32) -> Dict[str, any]:
        """评估模型性能
        
        Args:
            data: 评估数据，支持Tensor或字典格式
            batch_size: 批次大小，默认32
            
        Returns:
            包含评估结果的字典，通常包括：
            - loss: 损失值
            - accuracy: 准确率
            - perplexity: 困惑度
            
        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            RuntimeError: 当评估过程中出现错误时抛出
            
        Example:
            >>> metrics = model.evaluate(test_data)
            >>> print(f"Test accuracy: {metrics['accuracy']}")
        """
        pass

    @abstractmethod
    def predict(self,
               inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
               batch_size: int = 32,
               return_probs: bool = False) -> np.ndarray:
        """模型推理
        
        Args:
            inputs: 输入数据，支持Tensor或字典格式
            batch_size: 批次大小，默认32
            return_probs: 是否返回概率分布，默认False
            
        Returns:
            模型输出结果，numpy数组格式
            
        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            RuntimeError: 当推理过程中出现错误时抛出
            
        Example:
            >>> predictions = model.predict(test_inputs)
            >>> print(f"Predicted class: {np.argmax(predictions)}")
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """保存模型状态
        
        Args:
            path: 保存路径
            
        Raises:
            IOError: 当文件保存失败时抛出
            ValueError: 当路径无效时抛出
            
        Example:
            >>> model.save_model("model_state.pth")
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """加载模型状态
        
        Args:
            path: 加载路径
            
        Raises:
            IOError: 当文件加载失败时抛出
            ValueError: 当路径无效时抛出
            RuntimeError: 当模型加载失败时抛出
            
        Example:
            >>> model.load_model("model_state.pth")
        """
        pass

    @abstractmethod
    def generate_text(self,
                     prompt: str,
                     max_length: int = 50,
                     temperature: float = 0.7) -> str:
        """文本生成
        
        Args:
            prompt: 输入提示文本
            max_length: 生成文本最大长度，默认50
            temperature: 采样温度，控制生成多样性，默认0.7
            
        Returns:
            生成的文本字符串
            
        Raises:
            ValueError: 当输入提示为空时抛出
            RuntimeError: 当生成过程中出现错误时抛出
            
        Example:
            >>> generated = model.generate_text("量子计算的基本原理是")
            >>> print(generated)
        """
        pass

    @abstractmethod
    def initialize(self, model_config: Dict[str, Any]) -> None:
        """初始化模型
        
        Args:
            model_config: 包含模型配置的字典
            
        Raises:
            ValueError: 当配置无效时抛出
            RuntimeError: 当初始化失败时抛出
            
        Example:
            >>> config = {"type": "linear", "input_size": 10}
            >>> model.initialize(config)
        """
        pass
