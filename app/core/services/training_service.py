from typing import Any, Dict
from app.core.interfaces.training import ITrainingService

class TrainingService(ITrainingService):
    """训练服务类，实现ITrainingService接口
    
    该类作为训练服务的代理，将具体实现委托给_training_impl，
    提供训练流程的统一接口
    """
    def __init__(self, training_impl: ITrainingService):
        """初始化训练服务
        
        Args:
            training_impl (ITrainingService): 具体的训练服务实现
        """
        self._training_impl = training_impl  # 具体的训练服务实现

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """启动训练任务
        
        Args:
            config (Dict[str, Any]): 训练配置参数
            
        Returns:
            Dict[str, Any]: 训练启动结果
        """
        return self._training_impl.start_training(config)

    def stop_training(self) -> None:
        """停止训练任务"""
        self._training_impl.stop_training()

    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态
        
        Returns:
            Dict[str, Any]: 当前训练状态信息
        """
        return self._training_impl.get_training_status()

    def save_checkpoint(self, path: str) -> None:
        """保存训练检查点
        
        Args:
            path (str): 检查点保存路径
        """
        self._training_impl.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载训练检查点
        
        Args:
            path (str): 检查点文件路径
            
        Returns:
            Dict[str, Any]: 加载的检查点信息
        """
        return self._training_impl.load_checkpoint(path)

    def get_training_metrics(self) -> Dict[str, Any]:
        """获取训练指标
        
        Returns:
            Dict[str, Any]: 训练过程中的各项指标
        """
        return self._training_impl.get_training_metrics()
