from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import torch
import numpy as np

class IModelService(ABC):
    @abstractmethod
    def train(self,
             train_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
             val_data: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             early_stopping_patience: int = 5) -> Dict[str, any]:
        pass

    @abstractmethod
    def evaluate(self,
                data: Union[torch.Tensor, Dict[str, torch.Tensor]],
                batch_size: int = 32) -> Dict[str, any]:
        pass

    @abstractmethod
    def predict(self,
               inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
               batch_size: int = 32,
               return_probs: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        pass

    @abstractmethod
    def generate_text(self,
                     prompt: str,
                     max_length: int = 50,
                     temperature: float = 0.7) -> str:
        pass
