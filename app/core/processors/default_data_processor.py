"""
默认数据处理器实现

提供IDataProcessor接口的基本实现，用于在没有配置具体处理器时作为默认实现
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Union, Any
from app.core.interfaces.data_processing import IDataProcessor

class DefaultDataProcessor(IDataProcessor):
    """默认数据处理器实现类
    
    提供IDataProcessor接口的基本实现，主要用于：
    - 作为系统默认的数据处理器
    - 在没有配置具体处理器时使用
    - 提供最基本的数据处理功能
    """
    
    def __init__(self):
        """初始化默认数据处理器"""
        self.is_fitted = False
        self.metadata = {}
    
    def preprocess_data(self, 
                       raw_data: Union[pd.DataFrame, Dict[str, any]],
                       config: Dict[str, any] = None) -> Dict[str, np.ndarray]:
        """原始数据预处理
        
        Args:
            raw_data: 原始输入数据，支持DataFrame或字典格式
            config: 预处理配置参数，默认为None
            
        Returns:
            处理后的数据字典
        """
        if config is None:
            config = {}
            
        result = {}
        
        if isinstance(raw_data, pd.DataFrame):
            # 处理DataFrame格式数据
            for col in raw_data.columns:
                if pd.api.types.is_numeric_dtype(raw_data[col]):
                    # 数值型列直接转换为numpy数组
                    result[col] = raw_data[col].to_numpy()
                else:
                    # 非数值型列保持原样
                    result[col] = raw_data[col].to_numpy()
        elif isinstance(raw_data, dict):
            # 处理字典格式数据
            for key, value in raw_data.items():
                if isinstance(value, list):
                    result[key] = np.array(value)
                else:
                    result[key] = value
        else:
            raise ValueError(f"Unsupported data type: {type(raw_data)}")
            
        self.is_fitted = True
        return result

    def feature_engineering(self,
                           processed_data: Dict[str, np.ndarray],
                           feature_config: Dict[str, any] = None) -> Dict[str, np.ndarray]:
        """特征工程处理
        
        Args:
            processed_data: 预处理后的数据
            feature_config: 特征工程配置参数，默认为None
            
        Returns:
            包含特征工程结果的数据字典
        """
        if feature_config is None:
            feature_config = {}
            
        # 默认实现只是返回原始数据，不做额外处理
        return processed_data

    def split_data(self,
                  processed_data: Dict[str, np.ndarray],
                  split_config: Dict[str, any] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """数据集划分
        
        Args:
            processed_data: 处理后的完整数据集
            split_config: 数据集划分配置参数，默认为None
            
        Returns:
            划分后的数据集字典，包含train/val/test等key
        """
        if split_config is None:
            split_config = {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1
            }
            
        # 获取数据集大小
        data_size = 0
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                data_size = len(value)
                break
                
        if data_size == 0:
            return {
                "train": processed_data,
                "val": processed_data,
                "test": processed_data
            }
            
        # 计算各部分的索引
        train_size = int(data_size * split_config.get("train_ratio", 0.8))
        val_size = int(data_size * split_config.get("val_ratio", 0.1))
        
        train_indices = list(range(0, train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, data_size))
        
        # 划分数据集
        result = {
            "train": {},
            "val": {},
            "test": {}
        }
        
        for key, value in processed_data.items():
            if isinstance(value, np.ndarray):
                result["train"][key] = value[train_indices]
                result["val"][key] = value[val_indices]
                result["test"][key] = value[test_indices]
            else:
                result["train"][key] = value
                result["val"][key] = value
                result["test"][key] = value
                
        return result

    def save_processor(self, path: str) -> None:
        """保存处理器状态
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'is_fitted': self.is_fitted,
                'metadata': self.metadata
            }, f)

    def load_processor(self, path: str) -> None:
        """加载处理器状态
        
        Args:
            path: 加载路径
        """
        if not os.path.exists(path):
            raise IOError(f"Processor state file not found: {path}")
            
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.is_fitted = state.get('is_fitted', False)
            self.metadata = state.get('metadata', {})

    def load_dataset(self, path: str) -> Union[pd.DataFrame, Dict[str, any]]:
        """加载数据集
        
        Args:
            path: 数据集路径
            
        Returns:
            加载的数据集，支持DataFrame或字典格式
        """
        if not os.path.exists(path):
            raise IOError(f"Dataset file not found: {path}")
            
        # 根据文件扩展名决定加载方式
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.csv':
            return pd.read_csv(path)
        elif ext == '.json':
            return pd.read_json(path)
        elif ext == '.pkl' or ext == '.pickle':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif ext == '.npy':
            return {'data': np.load(path)}
        elif ext == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return {'text': lines}
        else:
            raise ValueError(f"Unsupported file format: {ext}") 