import pytest
from unittest.mock import Mock
from app.core.services.model_service import ModelService
from app.core.interfaces.model_operations import IModelOperations

class TestModelService:
    @pytest.fixture
    def mock_model_operator(self):
        mock = Mock()
        # 设置各个方法的返回值
        mock.load_model = Mock(return_value="test_model")
        mock.evaluate = Mock(return_value={"accuracy": 0.95})
        mock.predict = Mock(return_value=[0, 1, 0])
        mock.save_model = Mock()
        return mock

    @pytest.fixture
    def model_service(self, mock_model_operator):
        import torch
        device = torch.device('cpu')
        num_classes = 2
        return ModelService(
            model_ops=mock_model_operator
        )

    def test_load_model(self, model_service, mock_model_operator):
        # 测试模型加载
        result = model_service.load_model("model_path")
        
        mock_model_operator.load_model.assert_called_once_with("model_path")
        assert result == "test_model"

    def test_evaluate(self, model_service, mock_model_operator):
        # 测试模型评估
        test_data = [1, 2, 3]
        result = model_service.evaluate(test_data)
        
        mock_model_operator.evaluate.assert_called_once_with(test_data, batch_size=32)
        assert result == {"accuracy": 0.95}

    def test_predict(self, model_service, mock_model_operator):
        # 测试模型推理
        test_data = [1, 2, 3]
        result = model_service.predict(test_data)
        
        mock_model_operator.predict.assert_called_once_with(test_data, batch_size=32, return_probs=False)
        assert result == [0, 1, 0]

    def test_save_model(self, model_service, mock_model_operator):
        # 测试模型保存
        model_service.save_model("save_path")
        
        mock_model_operator.save_model.assert_called_once_with("save_path")
