import os
import torch
import threading
import psutil
import logging
import time
from datetime import datetime
from typing import Dict, List
from app.dao.model_dao import ModelDAO
from .core_operations import ModelCoreOperations

logger = logging.getLogger(__name__)

class ModelTrainingService(ModelCoreOperations):
    def __init__(self):
        super().__init__()
        self._monitoring = False
        self._monitor_thread = None
        

    def _cleanup_resources(self):
        """Clean up training resources"""
        try:
            logger.info('Cleaning up training resources')
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug('Cleared CUDA cache')
            
            # Delete temporary variables
            if hasattr(self, 'scaler'):
                del self.scaler
                logger.debug('Deleted gradient scaler')
            
            # Force garbage collection
            import gc
            gc.collect()
            logger.debug('Performed garbage collection')
            
            logger.info('Resource cleanup completed')
        except Exception as e:
            logger.error(f'Resource cleanup failed: {str(e)}')

    def _start_resource_monitoring(self, metrics):
        """Start resource monitoring thread"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(metrics,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info('Started resource monitoring thread')

    def _stop_resource_monitoring(self):
        """Stop resource monitoring thread"""
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
        """Monitor system resources during training"""
        try:
            while self._monitoring:
                # Collect CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                metrics['cpu_usage'].append(cpu_percent)
                
                # Collect memory usage
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                metrics['memory_usage'].append(memory_info.rss / 1024 / 1024)  # MB
                
                # Collect GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    metrics['gpu_memory'].append(gpu_memory)
                
                # Calculate throughput
                if len(metrics['epoch_times']) > 0:
                    throughput = len(metrics['loss']) / sum(metrics['epoch_times'])
                    metrics['throughput'].append(throughput)
                
                # Sleep to avoid excessive monitoring
                time.sleep(5)
        except Exception as e:
            logger.error(f'Resource monitoring failed: {str(e)}')
            self._monitoring = False
