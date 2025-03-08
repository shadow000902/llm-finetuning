from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class ITrainingService(ABC):
    """训练服务接口，定义模型训练的标准方法
    
    该接口定义了模型训练的核心操作，包括：
    - 启动/停止训练
    - 获取训练状态
    - 保存/加载检查点
    - 获取训练指标
    
    实现类需要提供具体的训练控制逻辑，适用于不同的训练框架
    """
    @abstractmethod
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """启动模型训练
        
        Args:
            config: 训练配置字典，包含：
                - model_config: 模型配置
                - data_config: 数据配置
                - optimizer_config: 优化器配置
                - training_params: 训练参数
                
        Returns:
            包含训练启动信息的字典，通常包括：
            - status: 启动状态
            - training_id: 训练ID
            - start_time: 开始时间
            
        Raises:
            ValueError: 当配置参数无效时抛出
            RuntimeError: 当训练启动失败时抛出
            
        Example:
            >>> config = {
                    "model_config": {...},
                    "data_config": {...},
                    "optimizer_config": {...},
                    "training_params": {...}
                }
            >>> result = trainer.start_training(config)
        """
        pass

    @abstractmethod
    def stop_training(self) -> None:
        """停止正在进行的训练
        
        Raises:
            RuntimeError: 当没有正在进行的训练时抛出
            RuntimeError: 当停止训练失败时抛出
            
        Example:
            >>> trainer.stop_training()
        """
        pass

    @abstractmethod
    def get_training_status(self) -> Dict[str, Any]:
        """获取当前训练状态
        
        Returns:
            包含训练状态信息的字典，通常包括：
            - status: 训练状态（running/stopped/completed）
            - progress: 训练进度
            - current_epoch: 当前epoch
            - elapsed_time: 已用时间
            
        Raises:
            RuntimeError: 当获取状态失败时抛出
            
        Example:
            >>> status = trainer.get_training_status()
            >>> print(f"Training progress: {status['progress']}%")
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """保存训练检查点
        
        Args:
            path: 检查点保存路径
            
        Raises:
            IOError: 当文件保存失败时抛出
            ValueError: 当路径无效时抛出
            
        Example:
            >>> trainer.save_checkpoint("checkpoints/epoch_10.pt")
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载训练检查点
        
        Args:
            path: 检查点加载路径
            
        Returns:
            包含加载信息的字典，通常包括：
            - status: 加载状态
            - loaded_epoch: 加载的epoch
            - model_state: 模型状态
            
        Raises:
            IOError: 当文件加载失败时抛出
            ValueError: 当路径无效时抛出
            RuntimeError: 当检查点加载失败时抛出
            
        Example:
            >>> checkpoint_info = trainer.load_checkpoint("checkpoints/epoch_10.pt")
        """
        pass

    @abstractmethod
    def get_training_metrics(self) -> Dict[str, Any]:
        """获取训练指标历史
        
        Returns:
            包含训练指标历史的字典，通常包括：
            - loss: 损失值历史
            - accuracy: 准确率历史
            - learning_rate: 学习率历史
            
        Raises:
            RuntimeError: 当获取指标失败时抛出
            
        Example:
            >>> metrics = trainer.get_training_metrics()
            >>> print(f"Final loss: {metrics['loss'][-1]}")
        """
        pass

    @abstractmethod
    def configure_training(self, config: Dict[str, Any]) -> None:
        """配置训练参数
        
        Args:
            config: 训练配置字典，包含：
                - epochs: 训练轮数
                - batch_size: 批次大小
                - learning_rate: 学习率
                
        Raises:
            ValueError: 当配置参数无效时抛出
        """
        pass

    @abstractmethod
    def execute_training(self, train_data: Any, val_data: Any) -> Dict[str, Any]:
        """执行模型训练
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            
        Returns:
            包含训练结果的字典，通常包括：
            - loss: 最终损失值
            - accuracy: 最终准确率
        """
        pass

    @abstractmethod
    def monitor_training(self) -> Dict[str, Any]:
        """监控训练过程
        
        Returns:
            包含训练指标的字典，通常包括：
            - loss: 当前损失值
            - accuracy: 当前准确率
        """
        pass

    @abstractmethod
    def save_training_results(self, results: Dict[str, Any], save_path: str) -> None:
        """保存训练结果
        
        Args:
            results: 训练结果字典
            save_path: 保存路径
        """
        pass
