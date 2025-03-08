from typing import Any, Dict, Union, Tuple, List
import pandas as pd
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

    def load_dataset(self, path: str) -> Union[pd.DataFrame, Dict[str, any]]:
        """加载数据集
        
        Args:
            path: 数据集路径
            
        Returns:
            加载的数据集，支持DataFrame或字典格式
            
        Raises:
            IOError: 当文件加载失败时抛出
            ValueError: 当路径无效或数据格式不支持时抛出
            
        Example:
            >>> dataset = processor.load_dataset("data.csv")
        """
        try:
            return self._processor.load_dataset(path)
        except FileNotFoundError as e:
            raise IOError(f"Failed to load dataset: {str(e)}")
        except Exception as e:
            raise ValueError(f"Invalid data format: {str(e)}")

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

    def feature_engineering(self, data: Any) -> Any:
        """特征工程处理
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的特征数据
            
        Raises:
            ValueError: 当输入数据格式无效时抛出
        """
        return self._processor.feature_engineering(data)

    def load_processor(self, path: str) -> None:
        """加载处理器状态
        
        Args:
            path: 处理器状态文件路径
        """
        self._processor.load_processor(path)

    def save_processor(self, path: str) -> None:
        """保存处理器状态
        
        Args:
            path: 处理器状态保存路径
        """
        self._processor.save_processor(path)

    def split_data(self, data: Any, ratios: List[float] = [0.8, 0.2]) -> Tuple[Any, Any]:
        """将数据集分割为训练集和验证集
        
        Args:
            data: 要分割的数据集
            ratios: 分割比例，默认为 [0.8, 0.2]，表示训练集占80%，验证集占20%
            
        Returns:
            (训练集, 验证集) 的元组
            
        Raises:
            ValueError: 当分割比例无效时抛出
            
        Example:
            >>> train_data, val_data = data_service.split_data(processed_data)
        """
        if not ratios:
            ratios = [0.8, 0.2]
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1")
        return self._processor.split_data(data, ratios)
