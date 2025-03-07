# 导入系统相关模块
import os
import threading
import time
import logging

# 导入机器学习相关模块
import torch

# 导入系统监控相关模块
import psutil

# 导入类型提示相关模块
from typing import Dict, List
from datetime import datetime

# 导入项目内部模块
from app.repositories.model_repository import ModelDAO
from .core_operations import ModelCoreOperations

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 模型训练服务类，继承自ModelCoreOperations
class ModelTrainingService(ModelCoreOperations):
    def __init__(self):
        """初始化模型训练服务"""
        super().__init__()
        self._monitoring = False  # 资源监控状态标志
        self._monitor_thread = None  # 资源监控线程
        

    def _cleanup_resources(self):
        """清理训练资源
        
        该方法用于在训练完成后释放系统资源，包括：
        - 清理CUDA缓存
        - 删除临时变量
        - 强制垃圾回收
        """
        try:
            logger.info('Cleaning up training resources')
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug('Cleared CUDA cache')
            
            # 删除临时变量
            if hasattr(self, 'scaler'):
                del self.scaler
                logger.debug('Deleted gradient scaler')
            
            # 强制垃圾回收
            import gc
            gc.collect()
            logger.debug('Performed garbage collection')
            
            logger.info('Resource cleanup completed')
        except Exception as e:
            logger.error(f'Resource cleanup failed: {str(e)}')

    def _start_resource_monitoring(self, metrics):
        """启动资源监控线程
        
        Args:
            metrics (dict): 用于存储监控指标的字典
        """
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,  # 设置监控函数
            args=(metrics,),  # 传入监控指标字典
            daemon=True  # 设置为守护线程
        )
        self._monitor_thread.start()
        logger.info('Started resource monitoring thread')

    def _stop_resource_monitoring(self):
        """停止资源监控线程
        
        该方法用于安全地停止资源监控线程，包括：
        - 设置监控状态标志为False
        - 等待监控线程结束
        - 处理线程未正常停止的情况
        """
        if hasattr(self, '_monitoring') and self._monitoring:
            self._monitoring = False
            if hasattr(self, '_monitor_thread'):
                self._monitor_thread.join(timeout=5)
                if self._monitor_thread.is_alive():
                    logger.warning('Resource monitoring thread did not stop gracefully')
                else:
                    logger.info('Stopped resource monitoring thread')
            else:
                logger.warning('No monitoring thread found to stop')
        else:
            logger.info('Resource monitoring was not running')

    def _monitor_resources(self, metrics):
        """监控系统资源使用情况
        
        Args:
            metrics (dict): 用于存储监控指标的字典
        """
        try:
            while self._monitoring:
                # 收集CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                metrics['cpu_usage'].append(cpu_percent)
                
                # 收集内存使用情况
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                metrics['memory_usage'].append(memory_info.rss / 1024 / 1024)  # 转换为MB
                
                # 如果GPU可用，收集GPU内存使用情况
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
                    metrics['gpu_memory'].append(gpu_memory)
                
                # 计算训练吞吐量
                if len(metrics['epoch_times']) > 0:
                    throughput = len(metrics['loss']) / sum(metrics['epoch_times'])
                    metrics['throughput'].append(throughput)
                
                # 休眠5秒以避免过度监控
                time.sleep(5)
        except Exception as e:
            logger.error(f'Resource monitoring failed: {str(e)}')
            self._monitoring = False
