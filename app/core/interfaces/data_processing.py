from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

class IDataProcessor(ABC):
    """数据处理器接口，定义数据预处理的标准操作
    
    该接口定义了数据预处理的通用操作流程，包括：
    - 数据预处理
    - 特征工程
    - 数据集划分
    - 处理器保存与加载
    
    实现类需要提供具体的数据处理逻辑，适用于不同的业务场景
    """
    @abstractmethod
    def preprocess_data(self, 
                       raw_data: Union[pd.DataFrame, Dict[str, any]],
                       config: Dict[str, any]) -> Dict[str, np.ndarray]:
        """原始数据预处理
        
        Args:
            raw_data: 原始输入数据，支持DataFrame或字典格式
            config: 预处理配置参数
            
        Returns:
            处理后的数据字典，key为数据标识，value为numpy数组
            
        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            KeyError: 当配置参数缺失必要字段时抛出
            
        Example:
            >>> processor = ConcreteDataProcessor()
            >>> data = {"text": ["sample1", "sample2"]}
            >>> config = {"max_length": 512}
            >>> processed = processor.preprocess_data(data, config)
        """
        pass

    @abstractmethod 
    def feature_engineering(self,
                           processed_data: Dict[str, np.ndarray],
                           feature_config: Dict[str, any]) -> Dict[str, np.ndarray]:
        """特征工程处理
        
        Args:
            processed_data: 预处理后的数据
            feature_config: 特征工程配置参数
            
        Returns:
            包含特征工程结果的数据字典
            
        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            KeyError: 当配置参数缺失必要字段时抛出
            
        Example:
            >>> features = processor.feature_engineering(
            ...     processed_data,
            ...     {"feature_type": "tfidf"}
            ... )
        """
        pass

    @abstractmethod
    def split_data(self,
                  processed_data: Dict[str, np.ndarray],
                  split_config: Dict[str, any]) -> Dict[str, np.ndarray]:
        """数据集划分
        
        Args:
            processed_data: 处理后的完整数据集
            split_config: 数据集划分配置参数
            
        Returns:
            划分后的数据集字典，通常包含train/val/test等key
            
        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            KeyError: 当配置参数缺失必要字段时抛出
            
        Example:
            >>> split_data = processor.split_data(
            ...     processed_data,
            ...     {"train_ratio": 0.8, "val_ratio": 0.1}
            ... )
        """
        pass

    @abstractmethod
    def save_processor(self, path: str) -> None:
        """保存处理器状态
        
        Args:
            path: 保存路径
            
        Raises:
            IOError: 当文件保存失败时抛出
            ValueError: 当路径无效时抛出
            
        Example:
            >>> processor.save_processor("processor_state.pkl")
        """
        pass

    @abstractmethod
    def load_processor(self, path: str) -> None:
        """加载处理器状态
        
        Args:
            path: 加载路径
            
        Raises:
            IOError: 当文件加载失败时抛出
            ValueError: 当路径无效时抛出
            KeyError: 当文件内容不完整时抛出
            
        Example:
            >>> processor.load_processor("processor_state.pkl")
        """
        pass
