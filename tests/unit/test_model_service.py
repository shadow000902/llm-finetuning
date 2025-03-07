import pytest
from unittest.mock import Mock
from app.core.services.model_service import ModelService
from app.core.interfaces.model_operations import IModelOperations

class TestModelService:
    @pytest.fixture
    def mock_model_operator(self):
        return Mock(spec=IModelOperations)

    @pytest.fixture
    def model_service(self, mock_model_operator):
        return ModelService(mock_model_operator)

    def test_load_model(self, model_service, mock_model_operator):
        # 测试模型加载
        test_model = "test_model"
        mock_model_operator.load_model.return_value = test_model
        
        result = model_service.load_model("model_path")
        
        mock_model_operator.load_model.assert_called_once_with("model_path")
        assert result == test_model

    def test_evaluate_model(self, model_service, mock_model_operator):
        # 测试模型评估
        test_data = [1, 2, 3]
        evaluation_result = {"accuracy": 0.95}
        mock_model_operator.evaluate_model.return_value = evaluation_result
        
        result = model_service.evaluate_model("model", test_data)
        
        mock_model_operator.evaluate_model.assert_called_once_with("model", test_data)
        assert result == evaluation_result

    def test_predict(self, model_service, mock_model_operator):
        # 测试模型推理
        test_data = [1, 2, 3]
        predictions = [0, 1, 0]
        mock_model_operator.predict.return_value = predictions
        
        result = model_service.predict("model", test_data)
        
        mock_model_operator.predict.assert_called_once_with("model", test_data)
        assert result == predictions

    def test_save_model(self, model_service, mock_model_operator):
        # 测试模型保存
        model_service.save_model("model", "save_path")
        
        mock_model_operator.save_model.assert_called_once_with("model", "save_path")
