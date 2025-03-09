from typing import Any, Dict
import logging
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
        self._trainer = training_impl  # 为了兼容测试用例

    def configure(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """配置训练参数（configure_training 的别名）
        
        Args:
            config: 训练配置参数
            
        Returns:
            Dict[str, Any]: 配置结果
        """
        return self.configure_training(config)

    def train(self,
             train_data: Any,
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             early_stopping_patience: int = 5) -> Dict[str, Any]:
        """训练模型
        
        Args:
            train_data: 训练数据
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            early_stopping_patience: 早停耐心值
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        return self.execute_training(train_data, None)

    def configure_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """配置训练参数
        
        Args:
            config: 训练配置参数
            
        Returns:
            Dict[str, Any]: 配置结果
        """
        return self._training_impl.configure_training(config)

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """启动训练任务
        
        Args:
            config: 训练配置参数
            
        Returns:
            Dict[str, Any]: 训练启动结果
            
        Raises:
            ValueError: 当配置参数无效时抛出
            RuntimeError: 当训练启动失败时抛出
        """
        try:
            return self._training_impl.start_training(config)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"启动训练失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"启动训练失败: {str(e)}")

    def stop_training(self) -> Dict[str, Any]:
        """停止训练任务
        
        Returns:
            Dict[str, Any]: 停止训练的结果
        """
        return self._training_impl.stop_training()

    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态
        
        Returns:
            Dict[str, Any]: 当前训练状态信息
        """
        return self._training_impl.get_training_status()

    def save_checkpoint(self, path: str) -> Dict[str, Any]:
        """保存训练检查点
        
        Args:
            path: 检查点保存路径
            
        Returns:
            Dict[str, Any]: 保存检查点的结果
        """
        return self._training_impl.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载训练检查点
        
        Args:
            path: 检查点文件路径
            
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

    def execute_training(self, train_data: Any, val_data: Any) -> Dict[str, Any]:
        """执行模型训练
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            
        Returns:
            Dict[str, Any]: 包含训练结果的字典
        """
        return self._training_impl.execute_training(train_data, val_data)

    def monitor_training(self) -> Dict[str, Any]:
        """监控训练过程
        
        Returns:
            Dict[str, Any]: 包含训练指标的字典
        """
        return self._training_impl.monitor_training()

    def save_training_results(self, results: Dict[str, Any], save_path: str) -> Dict[str, Any]:
        """保存训练结果
        
        Args:
            results: 训练结果字典
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 保存结果的状态
        """
        return self._training_impl.save_training_results(results, save_path)
