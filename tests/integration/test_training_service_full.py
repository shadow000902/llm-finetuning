import pytest
import os
import tempfile
import torch
import json
import threading
import time
from unittest.mock import Mock, patch, MagicMock
import sys

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.model.training_service import ModelTrainingService
from app.repositories.model_repository import ModelDAO

class TestModelTrainingServiceFull:
    @pytest.fixture
    def mock_model_dao(self):
        mock = MagicMock(spec=ModelDAO)
        mock.save_model_metadata = MagicMock(return_value=True)
        mock.get_model_metadata = MagicMock(return_value={"model_id": "test_model", "version": "1.0"})
        mock.update_model_status = MagicMock(return_value=True)
        mock.create_model_version = MagicMock()
        mock.get_model_version = MagicMock()
        return mock
    
    @pytest.fixture
    def training_service(self, mock_model_dao):
        with patch('app.model.training_service.ModelDAO', return_value=mock_model_dao):
            service = ModelTrainingService()
            service.loss = 0.0
            yield service
            if hasattr(service, '_monitor_thread') and service._monitor_thread is not None:
                service._monitoring = False
                service._monitor_thread.join(timeout=1)
    
    @pytest.fixture
    def sample_training_config(self):
        return {
            "model_name": "test_model",
            "learning_rate": 3e-5,
            "batch_size": 8,
            "epochs": 2,
            "max_length": 512,
            "warmup_steps": 100,
            "save_steps": 500,
            "output_dir": "./test_output"
        }
    
    @pytest.fixture
    def sample_dataset(self):
        return {
            "train": [{"input": "测试输入1", "output": "测试输出1"}, 
                     {"input": "测试输入2", "output": "测试输出2"}],
            "validation": [{"input": "验证输入1", "output": "验证输出1"}]
        }
    
    def test_initialize_training(self, training_service, sample_training_config):
        """测试训练初始化"""
        result = training_service.initialize_training(sample_training_config)
        
        assert result is not None
        assert "status" in result
        assert result["status"] == "initialized"
        assert "model_config" in result
    
    def test_start_resource_monitoring(self, training_service):
        """测试资源监控启动"""
        metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory': [],
            'throughput': []
        }
        
        training_service._start_resource_monitoring(metrics)
        assert training_service._monitoring is True
        assert training_service._monitor_thread is not None
        assert training_service._monitor_thread.is_alive()
        
        training_service._stop_resource_monitoring()
        assert training_service._monitoring is False
        training_service._monitor_thread.join(timeout=1)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1000000000)
    @patch('torch.cuda.memory_reserved', return_value=2000000000)
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent', return_value=50.0)
    def test_get_resource_usage(self, mock_cpu, mock_memory, mock_cuda_reserved, 
                              mock_cuda_allocated, mock_cuda_available, training_service):
        """测试获取资源使用情况"""
        metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory': [],
            'throughput': []
        }
        
        mock_memory_info = MagicMock()
        mock_memory_info.total = 16000000000
        mock_memory_info.used = 8000000000
        mock_memory.return_value = mock_memory_info
        
        result = training_service.get_resource_usage(metrics)
        
        assert len(metrics['cpu_usage']) > 0
        assert len(metrics['memory_usage']) > 0
        assert len(metrics['gpu_memory']) > 0
        assert "status" in result
    
    @patch('app.model.training_service.torch.save')
    def test_save_checkpoint(self, mock_save, training_service):
        """测试保存检查点"""
        training_service.model = Mock()
        training_service.optimizer = Mock()
        training_service.scheduler = Mock()
        training_service.current_epoch = 1
        training_service.loss = 0.5
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
            result = training_service.save_checkpoint(checkpoint_path)
            
            mock_save.assert_called_once()
            assert result is not None
            assert "status" in result
            assert result["status"] == "success"
            assert "checkpoint_path" in result
    
    @patch('app.model.training_service.torch.load')
    def test_load_checkpoint(self, mock_load, training_service):
        """测试加载检查点"""
        mock_checkpoint = {
            "model_state_dict": {"layer1.weight": torch.tensor([1.0])},
            "optimizer_state_dict": {"param_groups": []},
            "scheduler_state_dict": {},
            "epoch": 1,
            "loss": 0.5,
            "training_args": {"learning_rate": 3e-5}
        }
        mock_load.return_value = mock_checkpoint
        
        training_service.model = Mock()
        training_service.optimizer = Mock()
        training_service.scheduler = Mock()
        
        result = training_service.load_checkpoint("test_checkpoint.pt")
        
        mock_load.assert_called_once()
        training_service.model.load_state_dict.assert_called_once()
        training_service.optimizer.load_state_dict.assert_called_once()
        training_service.scheduler.load_state_dict.assert_called_once()
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert training_service.current_epoch == 1
        assert training_service.loss == 0.5
    
    @patch('app.model.training_service.ModelTrainingService._train_epoch')
    @patch('app.model.training_service.ModelTrainingService._validate')
    def test_train_model(self, mock_validate, mock_train_epoch, training_service, sample_dataset):
        """测试训练过程"""
        training_service.config = {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 3e-5,
            "output_dir": "./test_output"
        }
        
        mock_train_epoch.return_value = {"loss": 0.5, "perplexity": 1.5}
        mock_validate.return_value = {"val_loss": 0.4, "val_perplexity": 1.4}
        
        training_service.model = Mock()
        training_service.optimizer = Mock()
        training_service.scheduler = Mock()
        training_service.train_dataloader = Mock()
        training_service.val_dataloader = Mock()
        
        with patch('os.makedirs'):
            with patch('app.model.training_service.ModelTrainingService.save_checkpoint'):
                result = training_service.train_model("test_config.yaml")
        
        assert mock_train_epoch.call_count == 2
        assert mock_validate.call_count >= 1
        assert result is not None
        assert "status" in result
        assert result["status"] == "success"
        assert "metrics" in result
        assert "training_time" in result
