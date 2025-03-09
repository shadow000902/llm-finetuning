import pytest
import torch
import os
import tempfile
from unittest.mock import Mock, patch

from app.core.services.training_service import TrainingService
from app.core.interfaces.training import ITrainingService

class MockTrainingImplementation(ITrainingService):
    """用于测试的训练服务实现"""
    def __init__(self):
        self.config = None
        self.is_training = False
        self.checkpoint = None
        self.metrics = {"loss": 0.1, "accuracy": 0.95}
        self.results = None

    def configure_training(self, config):
        self.config = config
        return {"status": "configured", "config": config}
        
    def start_training(self, config):
        self.is_training = True
        return {"status": "started", "config": config}
        
    def stop_training(self):
        self.is_training = False
        return {"status": "stopped"}
        
    def get_training_status(self):
        status = "running" if self.is_training else "idle"
        return {"status": status}
        
    def save_checkpoint(self, path):
        self.checkpoint = path
        return {"status": "saved", "path": path}
        
    def load_checkpoint(self, path):
        self.checkpoint = path
        return {"status": "loaded", "path": path}
        
    def get_training_metrics(self):
        return self.metrics
        
    def execute_training(self, train_data, val_data=None):
        self.is_training = True
        # 模拟训练过程
        result = {
            "loss": 0.1, 
            "accuracy": 0.95, 
            "epochs": 10,
            "early_stopping": False
        }
        self.is_training = False
        return result
        
    def monitor_training(self):
        return {"loss": 0.1, "accuracy": 0.95, "progress": 0.5}
        
    def save_training_results(self, results, save_path):
        self.results = results
        return {"status": "saved", "path": save_path}

class TestTrainingServiceIntegration:
    @pytest.fixture
    def mock_training_implementation(self):
        return MockTrainingImplementation()

    @pytest.fixture
    def training_service(self, mock_training_implementation):
        return TrainingService(mock_training_implementation)

    @pytest.fixture
    def sample_train_data(self):
        # 创建示例训练数据
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
            "labels": torch.tensor([0, 1, 0])
        }

    @pytest.fixture
    def sample_val_data(self):
        # 创建示例验证数据
        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([1])
        }

    def test_training_service_initialization(self, training_service, mock_training_implementation):
        """测试训练服务初始化"""
        assert training_service._trainer == mock_training_implementation

    def test_configure_training(self, training_service, mock_training_implementation):
        """测试训练配置功能"""
        config = {"epochs": 10, "batch_size": 32, "learning_rate": 0.001}
        result = training_service.configure_training(config)
        
        # 验证配置被正确传递
        assert mock_training_implementation.config == config
        assert result["status"] == "configured"
        assert result["config"] == config

    def test_execute_training_workflow(self, training_service, mock_training_implementation, sample_train_data, sample_val_data):
        """测试完整训练工作流程"""
        # 1. 配置训练
        config = {"epochs": 10, "batch_size": 32, "learning_rate": 0.001}
        training_service.configure_training(config)
        
        # 2. 执行训练
        result = training_service.execute_training(sample_train_data, sample_val_data)
        
        # 3. 验证结果
        assert "loss" in result
        assert "accuracy" in result
        assert "epochs" in result
        assert result["loss"] == 0.1
        assert result["accuracy"] == 0.95

    def test_training_lifecycle(self, training_service, mock_training_implementation):
        """测试训练生命周期管理"""
        # 1. 开始训练
        config = {"epochs": 5}
        start_result = training_service.start_training(config)
        assert start_result["status"] == "started"
        assert mock_training_implementation.is_training == True
        
        # 2. 获取训练状态
        status = training_service.get_training_status()
        assert status["status"] == "running"
        
        # 3. 停止训练
        stop_result = training_service.stop_training()
        assert stop_result["status"] == "stopped"
        assert mock_training_implementation.is_training == False

    def test_checkpoint_management(self, training_service, mock_training_implementation):
        """测试检查点管理功能"""
        # 创建临时目录用于保存检查点
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            
            # 1. 保存检查点
            save_result = training_service.save_checkpoint(checkpoint_path)
            assert save_result["status"] == "saved"
            assert save_result["path"] == checkpoint_path
            assert mock_training_implementation.checkpoint == checkpoint_path
            
            # 2. 加载检查点
            load_result = training_service.load_checkpoint(checkpoint_path)
            assert load_result["status"] == "loaded"
            assert load_result["path"] == checkpoint_path

    def test_training_metrics_and_monitoring(self, training_service, mock_training_implementation):
        """测试训练指标和监控功能"""
        # 1. 获取训练指标
        metrics = training_service.get_training_metrics()
        assert metrics["loss"] == 0.1
        assert metrics["accuracy"] == 0.95
        
        # 2. 监控训练进度
        monitor_result = training_service.monitor_training()
        assert "loss" in monitor_result
        assert "accuracy" in monitor_result
        assert "progress" in monitor_result
        assert monitor_result["progress"] == 0.5

    def test_save_training_results(self, training_service, mock_training_implementation):
        """测试保存训练结果功能"""
        # 创建临时目录用于保存结果
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = os.path.join(temp_dir, "results.json")
            results = {"model": "model.pt", "metrics": {"accuracy": 0.95}}
            
            # 保存训练结果
            save_result = training_service.save_training_results(results, results_path)
            
            # 验证结果
            assert save_result["status"] == "saved"
            assert save_result["path"] == results_path
            assert mock_training_implementation.results == results