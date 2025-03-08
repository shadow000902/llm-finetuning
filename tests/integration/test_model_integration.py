import pytest
from unittest.mock import Mock

import torch

from app.core.services.data_service import DataService
from app.core.services.model_service import ModelService
from app.core.services.training_service import TrainingService
from app.core.interfaces.training import ITrainingService

class MockTrainingService(ITrainingService):
    def configure_training(self, config):
        return {"status": "configured"}
        
    def start_training(self, config):
        return {"status": "started"}
        
    def stop_training(self):
        pass
        
    def get_training_status(self):
        return {"status": "running"}
        
    def save_checkpoint(self, path):
        pass
        
    def load_checkpoint(self, path):
        return {"status": "loaded"}
        
    def get_training_metrics(self):
        return {"accuracy": 0.95}
        
    def execute_training(self, train_data, val_data):
        return {"accuracy": 0.95}
        
    def monitor_training(self):
        return {"loss": 0.1}
        
    def save_training_results(self, results, save_path):
        pass

class TestModelIntegration:
    @pytest.fixture
    def mock_processor(self):
        # Create mock tokenizer with required methods
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "mock text"
        mock_tokenizer.get_vocab.return_value = {"mock": 1}
        mock_tokenizer.mask_token_id = 4
        
        # Create real DataProcessing instance with mock tokenizer
        from app.core.services.model_components.data_processing import DataProcessing
        processor = DataProcessing(mock_tokenizer)
        
        # Mock specific methods while keeping real tokenizer
        processor.load_dataset = Mock(return_value=[1, 2, 3])
        processor.preprocess_data = Mock(return_value=[0.1, 0.2, 0.3])
        processor.split_data = Mock(return_value={"train": [0.1, 0.2], "val": [0.3]})
        return processor

    @pytest.fixture
    def services(self, mock_processor):
        # Create mock model and device
        mock_model = Mock()
        mock_model.load_model = Mock(return_value="test_model")
        mock_model.evaluate = Mock(return_value={"accuracy": 0.95})
        mock_model.predict = Mock(return_value=[0.1, 0.2, 0.3])
        device = torch.device('cpu')

        model_service = ModelService(
            model_ops=mock_model
        )

        return {
            "data_service": DataService(mock_processor),
            "model_service": model_service, 
            "training_service": TrainingService(MockTrainingService())
        }

    def test_model_lifecycle(self, services):
        # 测试模型完整生命周期
        # 1. 加载数据
        data = services["data_service"].load_dataset("test_data.csv")
        processed_data = services["data_service"].preprocess_data(data)
        train_data, val_data = services["data_service"].split_data(processed_data)
        
        # 2. 初始化模型
        model = services["model_service"].load_model({"type": "test_model"})
        
        # 3. 训练模型
        training_result = services["training_service"].execute_training(train_data, val_data)
        
        # 4. 评估模型
        eval_result = services["model_service"].evaluate(val_data)
        
        # 5. 保存模型
        services["model_service"].save_model("test_model.pth")
        
        # 验证结果
        assert "accuracy" in training_result
        assert "accuracy" in eval_result
        assert training_result["accuracy"] > 0
        assert eval_result["accuracy"] > 0

    def test_model_prediction(self, services):
        # 测试模型推理流程
        # 1. 加载模型
        model = services["model_service"].load_model({"type": "test_model"})
        
        # 2. 处理测试数据
        test_data = [1, 2, 3]
        processed_data = services["data_service"].preprocess_data(test_data)
        
        # 3. 执行预测
        predictions = services["model_service"].predict(processed_data)
        
        # 4. 验证结果
        assert len(predictions) == len(test_data)
        assert all(isinstance(p, float) for p in predictions)
