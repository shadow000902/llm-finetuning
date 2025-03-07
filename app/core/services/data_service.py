from typing import Any, Dict
from torch.utils.data import DataLoader
from app.core.interfaces.data_processing import IDataProcessor

class DataService(IDataProcessor):
    def __init__(self, processor: IDataProcessor):
        self._processor = processor

    def load_dataset(self, data_path: str) -> Any:
        return self._processor.load_dataset(data_path)

    def preprocess_data(self, data: Any) -> Any:
        return self._processor.preprocess_data(data)

    def create_data_loader(self, dataset: Any, batch_size: int) -> DataLoader:
        return self._processor.create_data_loader(dataset, batch_size)

    def get_data_stats(self) -> Dict[str, Any]:
        return self._processor.get_data_stats()
