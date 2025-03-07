import pytest
from unittest.mock import Mock
from app.core.services.data_service import DataService
from app.core.services.model_service import ModelService
from app.core.services.training_service import TrainingService

class TestModelIntegration:
    @pytest.fixture
    def mock_processor(self):
        processor = Mock()
        processor.load_dataset.return_value = "mock_dataset"
        processor.preprocess_data.return_value = "processed_data"
        processor.split_data.return_value = {"train": "train_data", "val": "val_data"}
        return processor

    @pytest.fixture
    def mock_processor(self):
        processor = Mock()
        processor.load_dataset.return_value = "mock_dataset"
        processor.preprocess_data.return_value = "processed_data"
        processor.split_data.return_value = {"train": "train_data", "val": "val_data"}
        return processor

    @pytest.fixture
    def services(self, mock_processor):
        return {
            "data_service": DataService(mock_processor),
            "model_service": ModelService(),
            "training_service": TrainingService()
        }

    def test_model_lifecycle(self, services):
        # 测试模型完整生命周期
        # 1. 加载数据
        data = services["data_service"].load_data("test_data.csv")
        processed_data = services["data_service"].preprocess_data(data)
        train_data, val_data = services["data_service"].split_data(processed_data)
        
        # 2. 初始化模型
        model = services["model_service"].load_model({"type": "test_model"})
        
        # 3. 训练模型
        training_result = services["training_service"].execute_training(train_data, val_data)
        
        # 4. 评估模型
        eval_result = services["model_service"].evaluate_model(model, val_data)
        
        # 5. 保存模型
        services["model_service"].save_model(model, "test_model.pth")
        
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
        predictions = services["model_service"].predict(model, processed_data)
        
        # 4. 验证结果
        assert len(predictions) == len(test_data)
        assert all(isinstance(p, float) for p in predictions)
