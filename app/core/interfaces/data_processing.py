from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

class IDataProcessor(ABC):
    @abstractmethod
    def preprocess_data(self, 
                       raw_data: Union[pd.DataFrame, Dict[str, any]],
                       config: Dict[str, any]) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod 
    def feature_engineering(self,
                           processed_data: Dict[str, np.ndarray],
                           feature_config: Dict[str, any]) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def split_data(self,
                  processed_data: Dict[str, np.ndarray],
                  split_config: Dict[str, any]) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def save_processor(self, path: str) -> None:
        pass

    @abstractmethod
    def load_processor(self, path: str) -> None:
        pass
