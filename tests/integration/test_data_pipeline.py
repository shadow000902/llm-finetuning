import pytest
import sys
import os
from unittest.mock import Mock

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.core.services.data_service import DataService
from app.core.services.model_service import ModelService
from app.core.services.training_service import TrainingService
from app.model.core_operations import ModelCoreOperations

class TestDataPipeline:
    @pytest.fixture
    def mock_processor(self):
        processor = Mock()
        processor.load_dataset.return_value = "mock_dataset"
        processor.preprocess_data.return_value = "processed_data"
        processor.split_data.return_value = {"train": "train_data", "val": "val_data"}
        return processor

    @pytest.fixture
    def core_operations(self, mock_processor):
        data_service = DataService(mock_processor)
        model_service = ModelService()
        training_service = TrainingService()
        return ModelCoreOperations(
            data_service=data_service,
            model_service=model_service,
            training_service=training_service
        )

    def test_end_to_end_training(self, core_operations):
        # 测试端到端训练流程
        # 1. 初始化模型
        model = core_operations.initialize_model({"type": "test_model"})
        
        # 2. 加载并处理数据
        data = core_operations.data_service.load_data("tests/test_data/test_data.csv")
        processed_data = core_operations.data_service.preprocess_data(data)
        train_data, val_data = core_operations.data_service.split_data(processed_data)
        
        # 3. 执行训练
        training_result = core_operations.train_model("tests/test_data/test_data.csv", {"epochs": 10})
        
        # 4. 验证结果
        assert "accuracy" in training_result
        assert training_result["accuracy"] > 0

    def test_end_to_end_prediction(self, core_operations):
        # 测试端到端推理流程
        # 1. 初始化模型
        model = core_operations.initialize_model({"type": "test_model"})
        
        # 2. 加载并处理测试数据
        test_data = [1, 2, 3]
        processed_data = core_operations.data_service.preprocess_data(test_data)
        
        # 3. 执行预测
        predictions = core_operations.predict(model, test_data)
        
        # 4. 验证结果
        assert len(predictions) == len(test_data)
        assert all(isinstance(p, int) for p in predictions)
