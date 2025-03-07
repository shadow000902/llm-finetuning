import os
import time
import threading
import logging
import psutil
import torch
import platform
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self, metrics: Dict[str, List[float]], interval: float = 1.0):
        self.metrics = metrics
        self.interval = interval
        self._monitoring = False
        self._monitor_thread = None
        self._start_time = None
        self._stop_event = threading.Event()
        self._history = {
            'cpu': deque(maxlen=1000),
            'memory': deque(maxlen=1000),
            'gpu': deque(maxlen=1000),
            'throughput': deque(maxlen=1000)
        }
        
    def start(self):
        """Start resource monitoring in background thread"""
        if self._monitoring:
            logger.warning('Monitoring already running')
            return
            
        self._monitoring = True
        self._start_time = time.time()
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info('Started resource monitoring')
        
    def stop(self):
        """Stop resource monitoring and clean up"""
        if not self._monitoring:
            logger.warning('Monitoring not running')
            return
            
        self._stop_event.set()
        self._monitor_thread.join(timeout=5)
        
        if self._monitor_thread.is_alive():
            logger.warning('Monitor thread did not stop cleanly')
            
        self._monitoring = False
        logger.info('Stopped resource monitoring')
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get current metrics snapshot"""
        return {
            'cpu': list(self._history['cpu']),
            'memory': list(self._history['memory']),
            'gpu': list(self._history['gpu']),
            'throughput': list(self._history['throughput'])
        }
        
    def _monitor_resources(self):
        """Main monitoring loop"""
        try:
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            
            while not self._stop_event.is_set():
                start_time = time.time()
                
                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self._history['cpu'].append(cpu_percent)
                self.metrics['cpu_usage'].append(cpu_percent)
                
                # Memory Usage
                memory_info = psutil.virtual_memory()
                self._history['memory'].append(memory_info.percent)
                self.metrics['memory_usage'].append(memory_info.percent)
                
                # GPU Usage
                if gpu_available:
                    gpu_memory = 0
                    gpu_utilization = 0
                    
                    for i in range(gpu_count):
                        mem_info = torch.cuda.memory_stats(i)
                        gpu_memory += mem_info['allocated_bytes.all.current']
                        gpu_utilization += torch.cuda.utilization(i)
                        
                    gpu_memory = gpu_memory / (1024 ** 3)  # Convert to GB
                    gpu_utilization = gpu_utilization / gpu_count
                    
                    self._history['gpu'].append(gpu_utilization)
                    self.metrics['gpu_memory'].append(gpu_memory)
                    self.metrics['gpu_utilization'].append(gpu_utilization)
                else:
                    self._history['gpu'].append(0)
                    self.metrics['gpu_memory'].append(0)
                    self.metrics['gpu_utilization'].append(0)
                    
                # Throughput Calculation
                if len(self.metrics['loss']) > 1:
                    time_diff = time.time() - self._start_time
                    samples_processed = len(self.metrics['loss'])
                    throughput = samples_processed / time_diff
                    self._history['throughput'].append(throughput)
                    self.metrics['throughput'].append(throughput)
                    
                # Sleep for interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f'Monitoring failed: {str(e)}', exc_info=True)
            self._monitoring = False
            
    def log_system_info(self):
        """Log detailed system information"""
        try:
            logger.info('System Information:')
            logger.info(f'- Platform: {platform.platform()}')
            logger.info(f'- Processor: {platform.processor()}')
            logger.info(f'- Python: {platform.python_version()}')
            logger.info(f'- PyTorch: {torch.__version__}')
            
            if torch.cuda.is_available():
                logger.info('CUDA Information:')
                for i in range(torch.cuda.device_count()):
                    logger.info(f'  Device {i}: {torch.cuda.get_device_name(i)}')
                    logger.info(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB')
                    logger.info(f'    Compute Capability: {torch.cuda.get_device_capability(i)}')
            else:
                logger.info('CUDA not available')
                
            # Log CPU info
            cpu_info = psutil.cpu_freq()
            if cpu_info:
                logger.info(f'- CPU Frequency: {cpu_info.current:.2f} MHz')
                logger.info(f'- CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical')
                
            # Log memory info
            mem_info = psutil.virtual_memory()
            logger.info(f'- Total Memory: {mem_info.total / (1024 ** 3):.2f} GB')
            logger.info(f'- Available Memory: {mem_info.available / (1024 ** 3):.2f} GB')
            
        except Exception as e:
            logger.error(f'Failed to log system info: {str(e)}')
            
    def generate_report(self) -> Dict[str, any]:
        """Generate resource usage report"""
        try:
            if not self._start_time:
                raise RuntimeError('Monitoring not started')
                
            duration = time.time() - self._start_time
            cpu_avg = sum(self._history['cpu']) / len(self._history['cpu']) if self._history['cpu'] else 0
            memory_avg = sum(self._history['memory']) / len(self._history['memory']) if self._history['memory'] else 0
            gpu_avg = sum(self._history['gpu']) / len(self._history['gpu']) if self._history['gpu'] else 0
            throughput_avg = sum(self._history['throughput']) / len(self._history['throughput']) if self._history['throughput'] else 0
            
            return {
                'duration': duration,
                'cpu_usage': cpu_avg,
                'memory_usage': memory_avg,
                'gpu_usage': gpu_avg,
                'throughput': throughput_avg,
                'metrics': self.metrics
            }
            
        except Exception as e:
            logger.error(f'Failed to generate report: {str(e)}')
            raise
