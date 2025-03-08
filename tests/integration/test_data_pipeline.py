import pytest
import sys
import os
import torch
from unittest.mock import Mock

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.core.services.data_service import DataService
from app.core.services.model_service import ModelService
from app.core.services.training_service import TrainingService
from app.model.core_operations import ModelCoreOperations
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

class TestDataPipeline:
    @pytest.fixture
    def mock_processor(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        from app.core.services.model_components.data_processing import DataProcessing
        processor = DataProcessing(tokenizer)
        processor.load_dataset = Mock(return_value="mock_dataset")
        processor.preprocess_data = Mock(return_value="processed_data")
        processor.split_data = Mock(return_value={"train": "train_data", "val": "val_data"})
        return processor

    @pytest.fixture
    def core_operations(self, mock_processor):
        data_service = DataService(mock_processor)
        
        # Create mock model and device
        mock_model = Mock()
        device = torch.device('cpu')
        
        model_service = ModelService(
            model_ops=mock_model
        )
        training_service = TrainingService(MockTrainingService())
        return ModelCoreOperations(
            data_service=data_service,
            model_service=model_service,
            training_service=training_service
        )

    def test_end_to_end_training(self, core_operations):
        # 测试端到端训练流程
        # 1. 初始化模型
        core_operations.initialize_model({"type": "test_model"})
        
        # 2. 加载并处理数据
        data = core_operations._data_service.load_dataset("tests/test_data/test_data.csv")
        processed_data = core_operations._data_service.preprocess_data(data)
        train_data, val_data = core_operations._data_service.split_data(processed_data)
        
        # 3. 执行训练
        training_result = core_operations.train_model(
            training_data=train_data,
            training_config={"epochs": 10}
        )
        
        # 4. 验证结果
        assert isinstance(training_result, dict)

    def test_end_to_end_prediction(self, core_operations):
        # 测试端到端推理流程
        # 1. 初始化模型
        model = core_operations.initialize_model({"type": "test_model"})
        
        # 2. 加载并处理测试数据
        test_data = [1, 2, 3]
        processed_data = core_operations._data_service.preprocess_data(test_data)
        
        # 3. 执行预测
        predictions = core_operations.predict(model, test_data)
        
        # 4. 验证结果
        assert predictions is not None  # 因为使用 mock，我们只能验证返回值不为 None
