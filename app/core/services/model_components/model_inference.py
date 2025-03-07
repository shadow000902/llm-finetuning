import os
import time
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self,
               inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
               batch_size: int = 32,
               return_probs: bool = False) -> np.ndarray:
        """Generate predictions for input data"""
        if isinstance(inputs, dict):
            dataset = inputs
        else:
            dataset = torch.utils.data.TensorDataset(inputs)
            
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                    
                batch = batch.to(self.device)
                outputs = self.model(batch)
                
                if return_probs:
                    if outputs.dim() > 1:
                        outputs = torch.softmax(outputs, dim=1)
                    else:
                        outputs = torch.sigmoid(outputs)
                        
                predictions.append(outputs.cpu())
                
        return torch.cat(predictions).numpy()
        
    def predict_classes(self,
                      inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                      batch_size: int = 32) -> np.ndarray:
        """Generate class predictions"""
        predictions = self.predict(inputs, batch_size)
        return np.argmax(predictions, axis=1)
        
    def predict_probs(self,
                    inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                    batch_size: int = 32) -> np.ndarray:
        """Generate probability predictions"""
        return self.predict(inputs, batch_size, return_probs=True)
        
    def predict_with_threshold(self,
                             inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                             threshold: float = 0.5,
                             batch_size: int = 32) -> np.ndarray:
        """Generate binary predictions using threshold"""
        probs = self.predict_probs(inputs, batch_size)
        return (probs > threshold).astype(int)
        
    def generate_prediction_report(self,
                                 inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                 targets: Optional[np.ndarray] = None,
                                 batch_size: int = 32) -> Dict[str, any]:
        """Generate comprehensive prediction report"""
        predictions = self.predict_classes(inputs, batch_size)
        probs = self.predict_probs(inputs, batch_size)
        
        report = {
            'predictions': predictions,
            'probabilities': probs,
            'input_shape': inputs.shape,
            'num_samples': len(inputs)
        }
        
        if targets is not None:
            if len(targets) != len(predictions):
                raise ValueError('Targets length does not match predictions')
                
            report['accuracy'] = np.mean(predictions == targets)
            report['correct_predictions'] = np.sum(predictions == targets)
            report['incorrect_predictions'] = np.sum(predictions != targets)
            
        return report
        
    def save_predictions(self,
                       inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                       output_path: str,
                       batch_size: int = 32) -> None:
        """Save predictions to file"""
        predictions = self.predict_classes(inputs, batch_size)
        np.save(output_path, predictions)
        logger.info(f'Saved predictions to {output_path}')

    def generate(self,
                input_ids: torch.Tensor,
                max_length: int = 100,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.95,
                repetition_penalty: float = 1.2,
                num_return_sequences: int = 1) -> List[str]:
        """Generate text sequences using the model"""
        try:
            if not isinstance(input_ids, torch.Tensor):
                raise ValueError('Input must be a torch.Tensor')
                
            if input_ids.dim() != 2:
                raise ValueError('Input tensor must be 2-dimensional')
                
            if max_length <= 0:
                raise ValueError('max_length must be positive')
                
            if temperature <= 0:
                raise ValueError('temperature must be positive')
                
            if top_k <= 0:
                raise ValueError('top_k must be positive')
                
            if not (0 < top_p <= 1):
                raise ValueError('top_p must be between 0 and 1')
                
            if repetition_penalty <= 0:
                raise ValueError('repetition_penalty must be positive')
                
            if num_return_sequences <= 0:
                raise ValueError('num_return_sequences must be positive')
                
            logger.info(f'Generating text with params: max_length={max_length}, '
                      f'temperature={temperature}, top_k={top_k}, top_p={top_p}, '
                      f'repetition_penalty={repetition_penalty}, '
                      f'num_return_sequences={num_return_sequences}')
                      
            input_ids = input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.model.config.pad_token_id,
                    eos_token_id=self.model.config.eos_token_id
                )
                
            return outputs.cpu()
            
        except Exception as e:
            logger.error(f'Text generation failed: {str(e)}')
            raise
