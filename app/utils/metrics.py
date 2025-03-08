"""
指标计算工具模块

该模块提供了用于计算和跟踪模型训练和评估指标的工具函数。
"""
import numpy as np
import torch
from typing import Dict, List, Union, Optional, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    用于跟踪和计算训练和评估指标的类
    """
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
    def update(self, metric_name: str, value: float) -> None:
        """
        更新指定指标的值
        
        Args:
            metric_name: 指标名称
            value: 指标值
        """
        if metric_name in self.metrics_history:
            self.metrics_history[metric_name].append(value)
            logger.debug(f"更新指标 {metric_name}: {value}")
        else:
            logger.warning(f"未知指标名称: {metric_name}")
            
    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        获取指定指标的最新值
        
        Args:
            metric_name: 指标名称
            
        Returns:
            最新的指标值，如果不存在则返回None
        """
        if metric_name in self.metrics_history and self.metrics_history[metric_name]:
            return self.metrics_history[metric_name][-1]
        return None
        
    def get_history(self, metric_name: str) -> List[float]:
        """
        获取指定指标的历史值
        
        Args:
            metric_name: 指标名称
            
        Returns:
            指标的历史值列表
        """
        return self.metrics_history.get(metric_name, [])
    
    def get_summary(self) -> Dict[str, float]:
        """
        获取所有指标的最新值
        
        Returns:
            包含所有指标最新值的字典
        """
        return {k: v[-1] if v else None for k, v in self.metrics_history.items()}


def calculate_perplexity(loss: float) -> float:
    """
    根据损失值计算困惑度
    
    Args:
        loss: 损失值
        
    Returns:
        困惑度值
    """
    return float(np.exp(loss))


def calculate_classification_metrics(
    predictions: Union[List[int], np.ndarray, torch.Tensor],
    labels: Union[List[int], np.ndarray, torch.Tensor],
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    计算分类任务的评估指标
    
    Args:
        predictions: 预测标签
        labels: 真实标签
        average: 平均方法，可选值为'micro', 'macro', 'weighted'
        
    Returns:
        包含准确率、精确率、召回率和F1分数的字典
    """
    # 确保输入是numpy数组
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def calculate_gpu_utilization() -> Dict[str, float]:
    """
    计算GPU利用率
    
    Returns:
        包含GPU内存使用率和计算单元使用率的字典
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_stats = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            memory_used_percent = memory_info.used / memory_info.total * 100
            gpu_stats.append({
                'device_id': i,
                'memory_used_percent': float(memory_used_percent),
                'gpu_utilization_percent': float(utilization.gpu)
            })
            
        return {'gpu_stats': gpu_stats}
    except (ImportError, Exception) as e:
        logger.warning(f"无法获取GPU利用率信息: {str(e)}")
        return {'gpu_stats': []}


def calculate_throughput(
    batch_size: int, 
    seq_length: int, 
    step_time: float
) -> float:
    """
    计算训练吞吐量（每秒处理的token数量）
    
    Args:
        batch_size: 批次大小
        seq_length: 序列长度
        step_time: 每步训练时间（秒）
        
    Returns:
        每秒处理的token数量
    """
    return (batch_size * seq_length) / step_time 