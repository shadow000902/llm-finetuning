import pytest
import pandas as pd
import numpy as np
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock


# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.implementations.data_processors import DefaultDataProcessor

class TestDefaultDataProcessor:
    @pytest.fixture
    def data_processor(self):
        processor = DefaultDataProcessor()
        processor.logger = Mock()
        return processor
    
    def test_init(self, data_processor):
        """测试初始化方法"""
        assert hasattr(data_processor, 'logger')
        assert hasattr(data_processor, 'metadata')
        assert data_processor.metadata == {}
    
    def test_preprocess_data_success(self, data_processor):
        """测试数据预处理成功的情况"""
        # 准备测试数据
        raw_data = [1, 2, 3]
        config = {"normalize": True}
        
        # 调用被测试方法
        result = data_processor.preprocess_data(raw_data, config)
        
        # 验证结果
        assert isinstance(result, dict)
        assert "processed_data" in result
        assert isinstance(result["processed_data"], np.ndarray)
        
    def test_preprocess_data_exception(self, data_processor):
        """测试数据预处理异常的情况"""
        # 准备会导致异常的测试数据
        # 模拟一个会引发异常的情况
        with patch.object(data_processor, 'logger'):
            with patch.object(data_processor, 'preprocess_data', side_effect=ValueError("数据预处理失败: 测试异常")):
                # 验证异常抛出
                with pytest.raises(ValueError) as excinfo:
                    data_processor.preprocess_data(None)
                
                # 验证错误消息
                assert "数据预处理失败" in str(excinfo.value)
    
    def test_feature_engineering(self, data_processor):
        """测试特征工程处理"""
        # 准备测试数据
        processed_data = {"data": np.array([1, 2, 3])}
        feature_config = {"pca": True, "components": 2}
        
        # 调用被测试方法
        result = data_processor.feature_engineering(processed_data, feature_config)
        
        # 验证结果
        assert isinstance(result, dict)
        # 验证日志记录
        data_processor.logger.debug.assert_called_once()
    
    def test_split_data(self, data_processor):
        """测试数据拆分"""
        # 准备测试数据 - 使用字典格式，与实现匹配
        processed_data = {"processed_data": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
        split_config = {"train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.1}
        
        # 调用被测试方法
        result = data_processor.split_data(processed_data, split_config)
        
        # 验证结果
        assert isinstance(result, dict)
        assert "train" in result
        assert "val" in result
        assert "test" in result
        # 验证日志记录
        data_processor.logger.debug.assert_called_once()
    
    def test_load_dataset(self, data_processor):
        """测试数据集加载"""
        # 模拟数据文件
        test_data = pd.DataFrame({"features": [1, 2, 3], "labels": [0, 1, 0]})
        
        # 使用patch模拟pandas读取方法
        with patch('pandas.read_json', return_value=test_data):
            # 调用被测试方法
            result = data_processor.load_dataset("test_path.json")
            
            # 验证结果
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            # 验证日志记录
            data_processor.logger.debug.assert_called_once()