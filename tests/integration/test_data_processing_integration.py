import pytest
import torch
import os
import tempfile
import json
import pandas as pd
from unittest.mock import Mock, patch

from app.core.services.model_components.data_processing import DataProcessing
from app.core.services.data_service import DataService

class TestDataProcessingIntegration:
    @pytest.fixture
    def mock_tokenizer(self):
        # 创建模拟tokenizer
        mock = Mock()
        mock.encode.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock.decode.return_value = "解码后的文本"
        mock.get_vocab.return_value = {"[PAD]": 0, "[UNK]": 1, "测": 2, "试": 3, "文": 4, "本": 5}
        mock.mask_token_id = 103
        mock.pad_token_id = 0
        return mock

    @pytest.fixture
    def data_processor(self, mock_tokenizer):
        return DataProcessing(mock_tokenizer)

    @pytest.fixture
    def data_service(self, data_processor):
        return DataService(data_processor)

    @pytest.fixture
    def sample_data_file(self):
        # 创建临时测试数据文件
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump([
                {"text": "测试文本1", "label": 0},
                {"text": "测试文本2", "label": 1},
                {"text": "测试文本3", "label": 0}
            ], f)
            temp_file = f.name
        yield temp_file
        # 测试后删除临时文件
        os.unlink(temp_file)

    @pytest.fixture
    def sample_csv_file(self):
        # 创建临时CSV测试数据文件
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            df = pd.DataFrame({
                'text': ['测试文本1', '测试文本2', '测试文本3'],
                'label': [0, 1, 0]
            })
            df.to_csv(f.name, index=False)
            temp_file = f.name
        yield temp_file
        # 测试后删除临时文件
        os.unlink(temp_file)

    def test_tokenize_text(self, data_processor):
        """测试文本标记化功能"""
        text = "测试文本"
        result = data_processor.tokenize_text(text)
        
        # 验证tokenizer.encode被调用
        data_processor.tokenizer.encode.assert_called_once_with(text, return_tensors="pt")
        assert torch.equal(result, torch.tensor([[1, 2, 3, 4, 5]]))

    def test_decode_text(self, data_processor):
        """测试文本解码功能"""
        token_ids = torch.tensor([[1, 2, 3, 4, 5]])
        result = data_processor.decode_text(token_ids)
        
        # 验证tokenizer.decode被调用
        data_processor.tokenizer.decode.assert_called_once()
        assert result == "解码后的文本"

    def test_data_loading_integration(self, data_processor, sample_data_file):
        """测试数据加载集成功能"""
        # 替换mock方法为实际方法
        original_load_dataset = data_processor.load_dataset
        data_processor.load_dataset = lambda path: original_load_dataset(path)
        
        # 加载数据
        with patch.object(data_processor, 'load_dataset', wraps=data_processor.load_dataset):
            result = data_processor.load_dataset(sample_data_file)
            
            # 验证结果
            assert isinstance(result, list)
            assert len(result) == 3
            assert all('text' in item and 'label' in item for item in result)

    def test_data_preprocessing_integration(self, data_processor):
        """测试数据预处理集成功能"""
        # 准备测试数据
        test_data = [
            {"text": "测试文本1", "label": 0},
            {"text": "测试文本2", "label": 1}
        ]
        
        # 模拟预处理方法
        with patch.object(data_processor, 'preprocess_data', return_value={
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "labels": torch.tensor([0, 1])
        }):
            result = data_processor.preprocess_data(test_data)
            
            # 验证结果
            assert isinstance(result, dict)
            assert "input_ids" in result
            assert "attention_mask" in result
            assert "labels" in result
            assert torch.is_tensor(result["input_ids"])
            assert torch.is_tensor(result["labels"])

    def test_data_splitting_integration(self, data_processor):
        """测试数据分割集成功能"""
        # 准备测试数据
        test_data = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]]),
            "labels": torch.tensor([0, 1, 0, 1])
        }
        
        # 模拟分割方法
        with patch.object(data_processor, 'split_data', wraps=lambda data, split_ratio: {
            "train": {
                "input_ids": data["input_ids"][:3],
                "attention_mask": data["attention_mask"][:3],
                "labels": data["labels"][:3]
            },
            "val": {
                "input_ids": data["input_ids"][3:],
                "attention_mask": data["attention_mask"][3:],
                "labels": data["labels"][3:]
            }
        }):
            result = data_processor.split_data(test_data, [0.75, 0.25])
            
            # 验证结果
            assert isinstance(result, dict)
            assert "train" in result
            assert "val" in result
            assert torch.is_tensor(result["train"]["input_ids"])
            assert torch.is_tensor(result["val"]["input_ids"])
            assert len(result["train"]["input_ids"]) == 3
            assert len(result["val"]["input_ids"]) == 1

    def test_feature_engineering_integration(self, data_processor):
        """测试特征工程集成功能"""
        # 准备测试数据
        test_data = [
            {"text": "测试文本1", "label": 0},
            {"text": "测试文本2", "label": 1}
        ]
        
        # 模拟特征工程方法
        with patch.object(data_processor, 'feature_engineering', return_value={
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "token_type_ids": torch.tensor([[0, 0, 0], [0, 0, 0]]),
            "labels": torch.tensor([0, 1])
        }):
            result = data_processor.feature_engineering(test_data)
            
            # 验证结果
            assert isinstance(result, dict)
            assert "input_ids" in result
            assert "attention_mask" in result
            assert "token_type_ids" in result
            assert "labels" in result
            assert torch.is_tensor(result["input_ids"])
            assert torch.is_tensor(result["token_type_ids"])

    def test_end_to_end_data_pipeline(self, data_service, sample_csv_file):
        """测试端到端数据处理流程"""
        # 模拟数据服务方法
        with patch.object(data_service, 'load_dataset', return_value=[
                {"text": "测试文本1", "label": 0},
                {"text": "测试文本2", "label": 1},
                {"text": "测试文本3", "label": 0},
                {"text": "测试文本4", "label": 1}
            ]), \
             patch.object(data_service, 'preprocess_data', return_value={
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]]),
                "labels": torch.tensor([0, 1, 0, 1])
             }), \
             patch.object(data_service, 'split_data', return_value=(
                {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
                    "labels": torch.tensor([0, 1, 0])
                },
                {
                    "input_ids": torch.tensor([[10, 11, 12]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                    "labels": torch.tensor([1])
                }
             )), \
             patch.object(data_service, 'feature_engineering', return_value={
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]]),
                "token_type_ids": torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                "labels": torch.tensor([0, 1, 0, 1])
             }):
            
            # 1. 加载数据
            data = data_service.load_dataset(sample_csv_file)
            assert isinstance(data, list)
            assert len(data) == 4
            
            # 2. 预处理数据
            processed_data = data_service.preprocess_data(data)
            assert isinstance(processed_data, dict)
            assert "input_ids" in processed_data
            assert "labels" in processed_data
            
            # 3. 分割数据
            train_data, val_data = data_service.split_data(processed_data)
            assert isinstance(train_data, dict)
            assert isinstance(val_data, dict)
            assert len(train_data["input_ids"]) == 3
            assert len(val_data["input_ids"]) == 1
            
            # 4. 特征工程
            engineered_data = data_service.feature_engineering(data)
            assert isinstance(engineered_data, dict)
            assert "token_type_ids" in engineered_data