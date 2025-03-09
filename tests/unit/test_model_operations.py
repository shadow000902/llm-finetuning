import pytest
import torch
from unittest.mock import Mock, patch
from app.implementations.model_operations import DefaultModelOperations

class TestDefaultModelOperations:
    @pytest.fixture
    def mock_model(self):
        mock = Mock()
        mock.to.return_value = mock
        mock.eval.return_value = mock
        mock.forward.return_value = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
        return mock

    @pytest.fixture
    def model_ops(self, mock_model):
        ops = DefaultModelOperations()
        ops.model = mock_model
        ops.logger = Mock()
        return ops

    def test_predict(self, model_ops):
        # 测试普通预测
        inputs = torch.randn(2, 3)
        result = model_ops.predict(inputs)
        assert isinstance(result, list)
        assert len(result) == 2
        model_ops.model.eval.assert_called_once()
        model_ops.model.forward.assert_called_once()

    def test_predict_with_probs(self, model_ops):
        # 测试带概率的预测
        inputs = torch.randn(2, 3)
        result = model_ops.predict(inputs, return_probs=True)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(p, list) and len(p) == 2 for p in result)

    @patch('torch.save')
    @patch('os.makedirs')
    def test_save_model(self, mock_makedirs, mock_save, model_ops):
        # 测试模型保存
        model_ops.save_model("test_path/model.pt")
        mock_makedirs.assert_called_once_with("test_path", exist_ok=True)
        mock_save.assert_called_once_with(model_ops.model, "test_path/model.pt")
        model_ops.logger.info.assert_called_once_with("Model saved to test_path/model.pt")

    def test_evaluate(self, model_ops):
        # 测试模型评估
        # 创建TensorDataset模拟真实数据格式
        test_inputs = torch.randn(2, 3)
        test_labels = torch.tensor([0, 1])
        import torch.utils.data as data
        test_data = data.TensorDataset(test_inputs, test_labels)
        
        result = model_ops.evaluate(test_data)
        
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result
        model_ops.logger.info.assert_any_call("Starting model evaluation")
        model_ops.logger.info.assert_any_call(
            "Evaluation complete - loss: {:.4f}, accuracy: {:.4f}".format(
                result["loss"], result["accuracy"]
            )
        )

    def test_evaluate_no_model(self, model_ops):
        # 测试无模型时的评估
        model_ops.model = None
        result = model_ops.evaluate(None)
        assert result == {"loss": None, "accuracy": None}
        model_ops.logger.warning.assert_called_once_with("No model available for evaluation")
