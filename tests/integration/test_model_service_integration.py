import pytest
import torch
from unittest.mock import Mock, patch

from app.core.services.model_service import ModelService
from app.core.interfaces.model_operations import IModelOperations

class TestModelServiceIntegration:
    @pytest.fixture
    def mock_model_operator(self):
        mock = Mock(spec=IModelOperations)
        # 设置各个方法的返回值
        mock.load_model = Mock(return_value="test_model")
        mock.evaluate = Mock(return_value={"accuracy": 0.95, "f1": 0.92, "precision": 0.94, "recall": 0.91})
        mock.predict = Mock(return_value=[0.1, 0.2, 0.7])
        mock.save_model = Mock()
        mock.generate_text = Mock(return_value="生成的测试文本")
        mock.model = Mock()
        mock.num_classes = 3
        mock.class_names = ["类别1", "类别2", "类别3"]
        mock.device = torch.device('cpu')
        return mock

    @pytest.fixture
    def model_service(self, mock_model_operator):
        return ModelService(model_ops=mock_model_operator)

    def test_model_service_initialization(self, model_service, mock_model_operator):
        """测试模型服务初始化"""
        assert model_service._model_ops == mock_model_operator

    def test_model_initialization(self, model_service, mock_model_operator):
        """测试模型初始化"""
        model_config = {"type": "bert", "pretrained": "bert-base-chinese"}
        model_service.initialize(model_config)
        
        # 验证initialize方法被调用
        if hasattr(mock_model_operator, 'initialize'):
            mock_model_operator.initialize.assert_called_once_with(model_config)

    def test_model_training(self, model_service, mock_model_operator):
        """测试模型训练"""
        # 准备训练数据
        train_data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        val_data = torch.tensor([[7, 8, 9]])
        
        # 设置mock返回值
        mock_model_operator.train.return_value = {
            "loss": 0.1, 
            "accuracy": 0.95,
            "epochs_completed": 10,
            "early_stopping_triggered": False
        }
        
        # 执行训练
        result = model_service.train(
            train_data=train_data,
            val_data=val_data,
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=5
        )
        
        # 验证train方法被正确调用
        mock_model_operator.train.assert_called_once_with(
            train_data=train_data,
            val_data=val_data,
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=5
        )
        
        # 验证返回结果
        assert result["loss"] == 0.1
        assert result["accuracy"] == 0.95
        assert result["epochs_completed"] == 10
        assert result["early_stopping_triggered"] == False

    def test_generate_text(self, model_service, mock_model_operator):
        """测试文本生成功能"""
        prompt = "这是一个测试提示"
        max_length = 100
        temperature = 0.8
        
        result = model_service.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        # 验证generate_text方法被正确调用
        mock_model_operator.generate_text.assert_called_once_with(
            prompt, 
            max_length=max_length, 
            temperature=temperature
        )
        
        assert result == "生成的测试文本"

    def test_get_model_info(self, model_service, mock_model_operator):
        """测试获取模型信息功能"""
        result = model_service.get_model_info()
        
        # 验证返回的模型信息
        assert "model_type" in result
        assert "num_classes" in result
        assert "class_names" in result
        assert "device" in result
        
        assert result["num_classes"] == 3
        assert result["class_names"] == ["类别1", "类别2", "类别3"]
        assert result["device"] == "cpu"

    def test_train_with_exception(self, model_service, mock_model_operator):
        """测试训练过程中的异常处理"""
        # 设置mock抛出异常
        mock_model_operator.train.side_effect = Exception("训练过程中出现错误")
        
        # 准备训练数据
        train_data = torch.tensor([[1, 2, 3]])
        
        # 验证异常被正确抛出和处理
        with pytest.raises(Exception) as excinfo:
            model_service.train(train_data=train_data)
            
        assert "训练过程中出现错误" in str(excinfo.value)