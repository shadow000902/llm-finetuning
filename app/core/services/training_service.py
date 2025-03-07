from typing import Any, Dict
from app.core.interfaces.training import ITrainingService

class TrainingService(ITrainingService):
    def __init__(self, training_impl: ITrainingService):
        self._training_impl = training_impl

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return self._training_impl.start_training(config)

    def stop_training(self) -> None:
        self._training_impl.stop_training()

    def get_training_status(self) -> Dict[str, Any]:
        return self._training_impl.get_training_status()

    def save_checkpoint(self, path: str) -> None:
        self._training_impl.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        return self._training_impl.load_checkpoint(path)

    def get_training_metrics(self) -> Dict[str, Any]:
        return self._training_impl.get_training_metrics()
