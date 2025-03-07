import os
import time
import logging
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.metrics_history = defaultdict(list)
        self._reset_batch_metrics()
        
    def _reset_batch_metrics(self):
        """Reset batch-level metrics"""
        self.batch_predictions = []
        self.batch_targets = []
        self.batch_losses = []
        
    def update_batch_metrics(self,
                           predictions: torch.Tensor,
                           targets: torch.Tensor,
                           loss: float) -> None:
        """Update metrics with batch results"""
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)
            
        self.batch_predictions.extend(predictions.cpu().numpy())
        self.batch_targets.extend(targets.cpu().numpy())
        self.batch_losses.append(loss)
        
    def calculate_epoch_metrics(self) -> Dict[str, float]:
        """Calculate and log metrics for completed epoch"""
        if not self.batch_predictions:
            raise ValueError('No predictions available for evaluation')
            
        predictions = np.array(self.batch_predictions)
        targets = np.array(self.batch_targets)
        loss = np.mean(self.batch_losses)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted'),
            'f1': f1_score(targets, predictions, average='weighted')
        }
        
        if self.num_classes == 2:
            metrics['roc_auc'] = roc_auc_score(targets, predictions)
            
        # Update metrics history
        for metric, value in metrics.items():
            self.metrics_history[metric].append(value)
            
        # Log confusion matrix
        cm = confusion_matrix(targets, predictions)
        self._log_confusion_matrix(cm)
        
        # Reset batch metrics for next epoch
        self._reset_batch_metrics()
        
        return metrics
        
    def _log_confusion_matrix(self, cm: np.ndarray) -> None:
        """Log confusion matrix with class names"""
        if len(self.class_names) != cm.shape[0]:
            logger.warning('Class names length does not match confusion matrix')
            return
            
        logger.info('Confusion Matrix:')
        header = ' ' * 10 + ' '.join(f'{name:^10}' for name in self.class_names)
        logger.info(header)
        
        for i, row in enumerate(cm):
            class_name = self.class_names[i]
            row_str = f'{class_name:>10} ' + ' '.join(f'{val:^10}' for val in row)
            logger.info(row_str)
            
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get complete metrics history"""
        return dict(self.metrics_history)
        
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics across all epochs"""
        best_metrics = {}
        
        for metric, values in self.metrics_history.items():
            if metric == 'loss':
                best_metrics[metric] = min(values)
            else:
                best_metrics[metric] = max(values)
                
        return best_metrics
        
    def early_stopping_check(self,
                            patience: int = 5,
                            min_delta: float = 0.01) -> bool:
        """Check if early stopping criteria is met"""
        if len(self.metrics_history['loss']) < patience + 1:
            return False
            
        recent_losses = self.metrics_history['loss'][-patience-1:]
        best_loss = min(recent_losses)
        current_loss = recent_losses[-1]
        
        if (current_loss - best_loss) > min_delta:
            return True
            
        return False
        
    def generate_evaluation_report(self) -> Dict[str, any]:
        """Generate comprehensive evaluation report"""
        report = {
            'best_metrics': self.get_best_metrics(),
            'metrics_history': self.get_metrics_history(),
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        return report
