import pytest
from app.core.services.data_service import DataService
from app.core.interfaces.data_processing import IDataProcessor

class TestDataService:
    @pytest.fixture
    def mock_data_processor(self, mocker):
        return mocker.Mock(spec=IDataProcessor)

    @pytest.fixture
    def data_service(self, mock_data_processor):
        return DataService(mock_data_processor)

    def test_load_dataset(self, data_service, mock_data_processor):
        # 测试数据集加载
        test_data = "test_dataset"
        mock_data_processor.load_dataset.return_value = test_data
        
        result = data_service.load_dataset("data_path")
        
        mock_data_processor.load_dataset.assert_called_once_with("data_path")
        assert result == test_data

    def test_preprocess_data(self, data_service, mock_data_processor):
        # 测试数据预处理
        test_data = "raw_data"
        processed_data = "processed_data"
        mock_data_processor.preprocess_data.return_value = processed_data
        
        result = data_service.preprocess_data(test_data)
        
        mock_data_processor.preprocess_data.assert_called_once_with(test_data)
        assert result == processed_data

    def test_split_data(self, data_service, mock_data_processor):
        # 测试数据拆分
        test_data = "full_data"
        split_data = {"train": "train_data", "val": "val_data"}
        mock_data_processor.split_data.return_value = split_data
        
        result = data_service.split_data(test_data, {"train": 0.8, "val": 0.2})
        
        mock_data_processor.split_data.assert_called_once_with(
            test_data, {"train": 0.8, "val": 0.2}
        )
        assert result == split_data

    def test_feature_engineering(self, data_service, mock_data_processor):
        # 测试特征工程
        test_data = "raw_features"
        processed_features = "engineered_features"
        mock_data_processor.feature_engineering.return_value = processed_features
        
        result = data_service.feature_engineering(test_data)
        
        mock_data_processor.feature_engineering.assert_called_once_with(test_data)
        assert result == processed_features
