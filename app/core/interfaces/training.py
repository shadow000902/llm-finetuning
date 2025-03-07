from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class ITrainingService(ABC):
    @abstractmethod
    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start model training with given configuration"""
        pass

    @abstractmethod
    def stop_training(self) -> None:
        """Stop ongoing training process"""
        pass

    @abstractmethod
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        pass

    @abstractmethod
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics history"""
        pass
