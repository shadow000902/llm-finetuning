from typing import Any, Dict
from torch.utils.data import DataLoader
from app.core.interfaces.data_processing import IDataProcessor

class DataService(IDataProcessor):
    """数据服务实现类，封装数据加载和预处理操作
    
    该类实现了IDataProcessor接口，提供了以下功能：
    - 数据集加载
    - 数据预处理
    - 数据加载器创建
    - 数据统计信息获取
    
    通过依赖注入的方式使用具体的数据处理器实现
    """
    def __init__(self, processor: IDataProcessor):
        """初始化数据服务
        
        Args:
            processor: 具体的数据处理器实现
        """
        self._processor = processor

    def load_dataset(self, data_path: str) -> Any:
        """加载数据集
        
        Args:
            data_path: 数据集路径
            
        Returns:
            加载的数据集对象
            
        Raises:
            FileNotFoundError: 当数据集路径不存在时抛出
            ValueError: 当数据集格式无效时抛出
            
        Example:
            >>> data_service = DataService(processor)
            >>> dataset = data_service.load_dataset("data/train.csv")
        """
        return self._processor.load_dataset(data_path)

    def preprocess_data(self, data: Any) -> Any:
        """预处理数据
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
            
        Raises:
            ValueError: 当输入数据格式无效时抛出
            
        Example:
            >>> processed_data = data_service.preprocess_data(raw_data)
        """
        return self._processor.preprocess_data(data)

    def create_data_loader(self, dataset: Any, batch_size: int) -> DataLoader:
        """创建数据加载器
        
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            
        Returns:
            DataLoader实例
            
        Raises:
            ValueError: 当数据集格式无效时抛出
            TypeError: 当批次大小类型错误时抛出
            
        Example:
            >>> loader = data_service.create_data_loader(dataset, batch_size=32)
        """
        return self._processor.create_data_loader(dataset, batch_size)

    def get_data_stats(self) -> Dict[str, Any]:
        """获取数据统计信息
        
        Returns:
            包含数据统计信息的字典，通常包括：
            - num_samples: 样本数量
            - feature_dim: 特征维度
            - class_distribution: 类别分布
            
        Raises:
            RuntimeError: 当无法获取统计信息时抛出
            
        Example:
            >>> stats = data_service.get_data_stats()
            >>> print(f"Number of samples: {stats['num_samples']}")
        """
        return self._processor.get_data_stats()
