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
    """模型评估器，用于评估模型性能并跟踪评估指标"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        初始化模型评估器
        
        Args:
            num_classes (int): 类别数量
            class_names (Optional[List[str]]): 类别名称列表，默认为数字编号
        """
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
        """
        更新批次评估指标
        
        Args:
            predictions (torch.Tensor): 模型预测结果
            targets (torch.Tensor): 真实标签
            loss (float): 当前批次的损失值
        """
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)
            
        self.batch_predictions.extend(predictions.cpu().numpy())
        self.batch_targets.extend(targets.cpu().numpy())
        self.batch_losses.append(loss)
        
    def calculate_epoch_metrics(self) -> Dict[str, float]:
        """
        计算并记录当前epoch的评估指标
        
        Returns:
            Dict[str, float]: 包含各项评估指标的字典，包括：
                - loss: 平均损失值
                - accuracy: 准确率
                - precision: 加权精确率
                - recall: 加权召回率
                - f1: 加权F1分数
                - roc_auc: 二分类时的ROC AUC值
        """
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
        """
        记录混淆矩阵
        
        Args:
            cm (np.ndarray): 混淆矩阵
        """
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
        """
        获取所有epoch的评估指标历史记录
        
        Returns:
            Dict[str, List[float]]: 包含各项指标历史记录的字典
        """
        return dict(self.metrics_history)
        
    def get_best_metrics(self) -> Dict[str, float]:
        """
        获取所有epoch中的最佳评估指标
        
        Returns:
            Dict[str, float]: 包含各项最佳指标的字典
        """
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
        """
        检查是否满足早停条件
        
        Args:
            patience (int): 允许性能不提升的epoch数，默认为5
            min_delta (float): 最小改进阈值，默认为0.01
            
        Returns:
            bool: 是否满足早停条件
        """
        if len(self.metrics_history['loss']) < patience + 1:
            return False
            
        recent_losses = self.metrics_history['loss'][-patience-1:]
        best_loss = min(recent_losses)
        current_loss = recent_losses[-1]
        
        if (current_loss - best_loss) > min_delta:
            return True
            
        return False
        
    def generate_evaluation_report(self) -> Dict[str, any]:
        """
        生成完整的评估报告
        
        Returns:
            Dict[str, any]: 包含以下内容的评估报告：
                - best_metrics: 最佳评估指标
                - metrics_history: 评估指标历史记录
                - num_classes: 类别数量
                - class_names: 类别名称
        """
        report = {
            'best_metrics': self.get_best_metrics(),
            'metrics_history': self.get_metrics_history(),
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        return report
