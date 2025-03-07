import pytest
from unittest.mock import Mock
from app.core.services.training_service import TrainingService
from app.core.interfaces.training import ITrainingService

class TestTrainingService:
    @pytest.fixture
    def mock_trainer(self):
        return Mock(spec=ITrainingService)

    @pytest.fixture
    def training_service(self, mock_trainer):
        return TrainingService(mock_trainer)

    def test_configure_training(self, training_service, mock_trainer):
        # 测试训练配置
        config = {"epochs": 10, "batch_size": 32}
        training_service.configure_training(config)
        
        mock_trainer.configure_training.assert_called_once_with(config)

    def test_execute_training(self, training_service, mock_trainer):
        # 测试训练执行
        train_data = [1, 2, 3]
        val_data = [4, 5]
        training_result = {"loss": 0.1, "accuracy": 0.95}
        mock_trainer.execute_training.return_value = training_result
        
        result = training_service.execute_training(train_data, val_data)
        
        mock_trainer.execute_training.assert_called_once_with(train_data, val_data)
        assert result == training_result

    def test_monitor_training(self, training_service, mock_trainer):
        # 测试训练监控
        metrics = {"loss": 0.1, "accuracy": 0.95}
        mock_trainer.monitor_training.return_value = metrics
        
        result = training_service.monitor_training()
        
        mock_trainer.monitor_training.assert_called_once()
        assert result == metrics

    def test_save_training_results(self, training_service, mock_trainer):
        # 测试训练结果保存
        results = {"model": "model.pth", "metrics": {"accuracy": 0.95}}
        training_service.save_training_results(results, "save_path")
        
        mock_trainer.save_training_results.assert_called_once_with(results, "save_path")
