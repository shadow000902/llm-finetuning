import pytest
import torch
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.model.implementations.deepseek_model import DeepSeekModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.core.interfaces.model_operations import IModelOperations

class TestDeepSeekModel:
    @pytest.fixture
    def mock_tokenizer(self):
        mock = Mock(spec=AutoTokenizer)
        mock.pad_token_id = 0
        mock.eos_token_id = 2
        mock.encode.return_value = torch.tensor([[1, 2, 3]])
        mock.decode.return_value = "测试文本"
        mock.batch_encode_plus.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        return mock
    
    @pytest.fixture
    def mock_model(self):
        mock = Mock(spec=AutoModelForCausalLM)
        mock.config.vocab_size = 32000
        mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock.to.return_value = mock  # 确保to()方法返回模型自身
        return mock
    
    @pytest.fixture
    def deepseek_model(self):
        with patch('app.model.implementations.deepseek_model.AutoModelForCausalLM') as mock_auto_model:
            with patch('app.model.implementations.deepseek_model.AutoTokenizer') as mock_auto_tokenizer:
                # 使用MagicMock来模拟抽象类的实现
                with patch.multiple(DeepSeekModel, __abstractmethods__=set()):
                    model = DeepSeekModel(model_path="test/model/path", device="cpu")
                    # 添加必要的抽象方法实现
                    model.predict = MagicMock()
                    model.generate_text = MagicMock()
                    model.train = MagicMock()
                    model.evaluate = MagicMock()
                    yield model
    
    def test_init(self, deepseek_model):
        """测试初始化方法"""
        assert deepseek_model.model_path == "test/model/path"
        assert deepseek_model.device == "cpu"
        assert deepseek_model.model is None
        assert deepseek_model.tokenizer is None
    
    @patch('app.model.implementations.deepseek_model.AutoModelForCausalLM.from_pretrained')
    @patch('app.model.implementations.deepseek_model.AutoTokenizer.from_pretrained')
    def test_load_model(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained, deepseek_model):
        """测试模型加载方法"""
        # 设置模拟返回值
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        
        # 调用被测试方法
        deepseek_model.load_model()
        
        # 验证方法调用
        mock_model_from_pretrained.assert_called_once_with(
            "test/model/path",
            device_map="cpu",
            trust_remote_code=True
        )
        mock_tokenizer_from_pretrained.assert_called_once_with(
            "test/model/path",
            trust_remote_code=True
        )
        
        # 验证属性设置
        assert deepseek_model.model == mock_model
        assert deepseek_model.tokenizer == mock_tokenizer
    
    @patch('app.model.implementations.deepseek_model.get_peft_model')
    @patch('app.model.implementations.deepseek_model.prepare_model_for_kbit_training')
    @patch('app.model.implementations.deepseek_model.LoraConfig')
    def test_prepare_for_training(self, mock_lora_config, mock_prepare, mock_get_peft, deepseek_model):
        """测试训练准备方法"""
        # 设置模型和tokenizer
        deepseek_model.model = Mock()
        deepseek_model.tokenizer = Mock()
        mock_prepare.return_value = deepseek_model.model
        mock_get_peft.return_value = Mock()
        mock_lora_config.return_value = Mock()
        
        # 调用被测试方法
        config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        deepseek_model.prepare_for_training(config)
        
        # 验证方法调用
        mock_prepare.assert_called_once()
        mock_get_peft.assert_called_once()
        mock_lora_config.assert_called_once_with(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        deepseek_model.model.print_trainable_parameters.assert_called_once()
    
    def test_tokenize_text(self, deepseek_model):
        """测试文本标记化方法"""
        # 设置tokenizer
        deepseek_model.tokenizer = Mock()
        deepseek_model.tokenizer.encode.return_value = torch.tensor([[1, 2, 3, 4]])
        
        # 调用被测试方法
        result = deepseek_model.tokenize_text("测试文本")
        
        # 验证方法调用和返回值
        deepseek_model.tokenizer.encode.assert_called_once_with("测试文本", return_tensors="pt")
        assert isinstance(result, torch.Tensor)
    
    def test_generate_text(self, deepseek_model):
        """测试文本生成方法"""
        # 设置模型和tokenizer
        deepseek_model.model = Mock()
        deepseek_model.tokenizer = Mock()
        deepseek_model.tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        deepseek_model.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        deepseek_model.tokenizer.decode.return_value = "生成的文本"
        
        # 调用被测试方法
        result = deepseek_model.generate_text("测试输入", max_length=50)
        
        # 验证方法调用和返回值
        deepseek_model.tokenizer.encode.assert_called_once_with("测试输入", return_tensors="pt")
        deepseek_model.model.generate.assert_called_once()
        deepseek_model.tokenizer.decode.assert_called_once_with(torch.tensor([[1, 2, 3, 4, 5]]))
        assert result == "生成的文本"
    
    def test_save_model(self, deepseek_model):
        """测试模型保存方法"""
        # 设置模型和tokenizer
        deepseek_model.model = Mock()
        deepseek_model.tokenizer = Mock()
        
        # 调用被测试方法
        with patch('os.makedirs') as mock_makedirs:
            deepseek_model.save_model("test/save/path")
        
        # 验证方法调用
        mock_makedirs.assert_called_once()
        deepseek_model.model.save_pretrained.assert_called_once_with("test/save/path")
        deepseek_model.tokenizer.save_pretrained.assert_called_once_with("test/save/path")
