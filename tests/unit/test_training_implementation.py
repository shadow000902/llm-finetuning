import pytest
import torch
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.implementations.training import DefaultTrainingService
from app.core.interfaces.training import ITrainingService

class TestDefaultTrainingService:
    @pytest.fixture
    def training_implementation(self):
        with patch('app.implementations.training.DefaultTrainingService.logger', new=Mock()) as mock_logger:
            implementation = DefaultTrainingService()
            return implementation
    
    def test_init(self, training_implementation):
        """测试初始化方法"""
        assert hasattr(training_implementation, 'logger')
        assert training_implementation.model is None
        assert training_implementation.optimizer is None
        assert training_implementation.scheduler is None
        assert training_implementation.config == {}
        assert training_implementation.current_epoch == 0
        assert training_implementation.global_step == 0
    
    def test_configure_training(self, training_implementation):
        """测试训练配置方法"""
        # 准备测试数据
        config = {
            "learning_rate": 3e-5,
            "batch_size": 16,
            "epochs": 3,
            "warmup_steps": 100
        }
        
        # 调用被测试方法
        result = training_implementation.configure_training(config)
        
        # 验证结果
        assert training_implementation.config == config
        assert result["status"] == "configured"
    
    @patch('torch.optim.AdamW')
    def test_setup_optimizer(self, mock_adamw, training_implementation):
        """测试优化器设置方法"""
        # 设置模型和配置
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
        training_implementation.model = mock_model
        training_implementation.config = {"learning_rate": 3e-5}
        
        # 调用被测试方法
        training_implementation._setup_optimizer()
        
        # 验证方法调用
        mock_adamw.assert_called_once()
        assert training_implementation.optimizer is not None
    
    @patch('torch.optim.lr_scheduler.LambdaLR')
    def test_setup_scheduler(self, mock_scheduler, training_implementation):
        """测试学习率调度器设置方法"""
        # 设置优化器和配置
        training_implementation.optimizer = Mock()
        training_implementation.config = {"warmup_steps": 100, "epochs": 3}
        
        # 调用被测试方法
        training_implementation._setup_scheduler()
        
        # 验证方法调用
        mock_scheduler.assert_called_once()
        assert training_implementation.scheduler is not None
    
    def test_start_training(self, training_implementation):
        """测试训练启动方法"""
        # 设置配置
        config = {"learning_rate": 3e-5, "batch_size": 16}
        
        # 使用patch模拟内部方法
        with patch.object(training_implementation, '_setup_model') as mock_setup_model:
            with patch.object(training_implementation, '_setup_optimizer') as mock_setup_optimizer:
                with patch.object(training_implementation, '_setup_scheduler') as mock_setup_scheduler:
                    # 调用被测试方法
                    result = training_implementation.start_training(config)
                    
                    # 验证方法调用和返回值
                    mock_setup_model.assert_called_once()
                    mock_setup_optimizer.assert_called_once()
                    mock_setup_scheduler.assert_called_once()
                    assert result["status"] == "started"
                    assert result["config"] == config
    
    def test_stop_training(self, training_implementation):
        """测试训练停止方法"""
        # 设置训练状态
        training_implementation.is_training = True
        
        # 调用被测试方法
        result = training_implementation.stop_training()
        
        # 验证结果
        assert training_implementation.is_training is False
        assert result["status"] == "stopped"
    
    def test_get_training_status(self, training_implementation):
        """测试获取训练状态方法"""
        # 设置训练状态和指标
        training_implementation.is_training = True
        training_implementation.current_epoch = 2
        training_implementation.global_step = 100
        training_implementation.metrics = {"loss": 0.5, "accuracy": 0.8}
        
        # 调用被测试方法
        result = training_implementation.get_training_status()
        
        # 验证结果
        assert result["status"] == "running"
        assert result["epoch"] == 2
        assert result["step"] == 100
        assert result["metrics"] == {"loss": 0.5, "accuracy": 0.8}
    
    @patch('torch.save')
    def test_save_checkpoint(self, mock_save, training_implementation):
        """测试保存检查点方法"""
        # 设置模型、优化器和调度器
        training_implementation.model = Mock()
        training_implementation.optimizer = Mock()
        training_implementation.scheduler = Mock()
        training_implementation.current_epoch = 2
        training_implementation.global_step = 100
        
        # 使用patch模拟目录创建
        with patch('os.makedirs') as mock_makedirs:
            with patch('os.path.dirname', return_value="test_dir") as mock_dirname:
                # 调用被测试方法
                result = training_implementation.save_checkpoint("test_path.pt")
                
                # 验证方法调用和返回值
                mock_dirname.assert_called_once_with("test_path.pt")
                mock_makedirs.assert_called_once_with("test_dir", exist_ok=True)
                mock_save.assert_called_once()
                assert result["status"] == "saved"
                assert result["path"] == "test_path.pt"
    
    @patch('torch.load')
    def test_load_checkpoint(self, mock_load, training_implementation):
        """测试加载检查点方法"""
        # 设置模型、优化器和调度器
        training_implementation.model = Mock()
        training_implementation.optimizer = Mock()
        training_implementation.scheduler = Mock()
        
        # 模拟加载的检查点数据
        checkpoint_data = {
            "model_state_dict": {"layer1.weight": torch.tensor([1.0])},
            "optimizer_state_dict": {"param_groups": []},
            "scheduler_state_dict": {},
            "epoch": 2,
            "global_step": 100,
            "config": {"learning_rate": 3e-5}
        }
        mock_load.return_value = checkpoint_data
        
        # 调用被测试方法
        result = training_implementation.load_checkpoint("test_path.pt")
        
        # 验证方法调用和返回值
        mock_load.assert_called_once_with("test_path.pt", map_location="cpu")
        training_implementation.model.load_state_dict.assert_called_once()
        training_implementation.optimizer.load_state_dict.assert_called_once()
        training_implementation.scheduler.load_state_dict.assert_called_once()
        assert training_implementation.current_epoch == 2
        assert training_implementation.global_step == 100
        assert training_implementation.config == {"learning_rate": 3e-5}
        assert result["status"] == "loaded"
        assert result["path"] == "test_path.pt"