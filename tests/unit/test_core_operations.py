import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.model.core_operations import ModelCoreOperations
from unittest.mock import Mock
from app.core.interfaces.model_operations import IModelService, ITrainingService
from app.core.interfaces.data_processing import IDataProcessor

class TestCoreOperations:
    @pytest.fixture
    def mock_services(self):
        return {
            "model_service": Mock(spec=IModelService),
            "training_service": Mock(spec=ITrainingService),
            "data_service": Mock(spec=IDataProcessor)
        }

    @pytest.fixture
    def core_operations(self, mock_services):
        return ModelCoreOperations(
            mock_services["model_service"],
            mock_services["training_service"],
            mock_services["data_service"]
        )

    def test_initialize_model(self, core_operations, mock_services):
        # 测试模型初始化
        model_config = {"type": "linear", "input_size": 10}
        core_operations.initialize_model(model_config)
        
        mock_services["model_service"].initialize.assert_called_once_with(model_config)

    @pytest.fixture
    def mock_data_processor(self):
        return Mock(spec=IDataProcessor)

    def test_train_model(self, core_operations, mock_services):
        # 测试模型训练
        training_data = [1, 2, 3]
        training_config = {"epochs": 10}
        mock_services["data_service"].preprocess_data.return_value = training_data
        mock_services["training_service"].train.return_value = {"loss": 0.1}
        
        result = core_operations.train_model(training_data, training_config)
        
        mock_services["data_service"].preprocess_data.assert_called_once_with(training_data)
        mock_services["training_service"].train.assert_called_once_with(
            train_data=training_data,
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=5
        )
        assert result == {"loss": 0.1}

    def test_predict(self, core_operations, mock_services):
        # 测试模型预测
        model = "test_model"
        input_data = [1, 2, 3]
        processed_data = [0.1, 0.2, 0.3]
        mock_services["data_service"].preprocess_data.return_value = processed_data
        mock_services["model_service"].predict.return_value = 1.0
        
        result = core_operations.predict(model, input_data)
        
        mock_services["data_service"].preprocess_data.assert_called_once_with(input_data)
        mock_services["model_service"].predict.assert_called_once_with(processed_data)
        assert result == 1.0
