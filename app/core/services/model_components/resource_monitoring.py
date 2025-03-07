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
    """资源监控类，用于监控系统资源使用情况
    
    该类可以监控CPU、内存、GPU使用率以及模型训练吞吐量等指标，
    并将监控数据存储在历史记录中以便后续分析
    """
    def __init__(self, metrics: Dict[str, List[float]], interval: float = 1.0):
        """初始化资源监控器
        
        Args:
            metrics (Dict[str, List[float]]): 用于存储监控指标的字典
            interval (float, optional): 监控间隔时间，单位秒. Defaults to 1.0.
        """
        self.metrics = metrics  # 存储各项监控指标的字典
        self.interval = interval  # 监控采样间隔时间
        self._monitoring = False  # 监控是否正在运行
        self._monitor_thread = None  # 监控线程
        self._start_time = None  # 监控开始时间
        self._stop_event = threading.Event()  # 用于停止监控的事件
        self._history = {
            'cpu': deque(maxlen=1000),  # 存储CPU使用率历史数据
            'memory': deque(maxlen=1000),  # 存储内存使用率历史数据
            'gpu': deque(maxlen=1000),  # 存储GPU使用率历史数据
            'throughput': deque(maxlen=1000)  # 存储吞吐量历史数据
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
        """主监控循环，定期采集系统资源使用情况"""
        try:
            gpu_available = torch.cuda.is_available()  # 检查GPU是否可用
            gpu_count = torch.cuda.device_count() if gpu_available else 0  # 获取GPU数量
            
            while not self._stop_event.is_set():
                start_time = time.time()  # 记录循环开始时间
                
                # CPU使用率监控
                cpu_percent = psutil.cpu_percent(interval=None)  # 获取CPU使用率
                self._history['cpu'].append(cpu_percent)  # 记录到历史数据
                self.metrics['cpu_usage'].append(cpu_percent)  # 记录到指标字典
                
                # 内存使用率监控
                memory_info = psutil.virtual_memory()  # 获取内存信息
                self._history['memory'].append(memory_info.percent)  # 记录内存使用率
                self.metrics['memory_usage'].append(memory_info.percent)  # 记录到指标字典
                
                # GPU使用率监控
                if gpu_available:
                    gpu_memory = 0  # GPU显存使用量
                    gpu_utilization = 0  # GPU利用率
                    
                    # 遍历所有GPU设备
                    for i in range(gpu_count):
                        mem_info = torch.cuda.memory_stats(i)  # 获取显存信息
                        gpu_memory += mem_info['allocated_bytes.all.current']  # 累加显存使用量
                        gpu_utilization += torch.cuda.utilization(i)  # 累加GPU利用率
                        
                    gpu_memory = gpu_memory / (1024 ** 3)  # 将显存使用量转换为GB
                    gpu_utilization = gpu_utilization / gpu_count  # 计算平均GPU利用率
                    
                    self._history['gpu'].append(gpu_utilization)  # 记录GPU利用率
                    self.metrics['gpu_memory'].append(gpu_memory)  # 记录显存使用量
                    self.metrics['gpu_utilization'].append(gpu_utilization)  # 记录GPU利用率
                else:
                    self._history['gpu'].append(0)  # 无GPU时记录0
                    self.metrics['gpu_memory'].append(0)  # 无GPU时记录0
                    self.metrics['gpu_utilization'].append(0)  # 无GPU时记录0
                    
                # 吞吐量计算
                if len(self.metrics['loss']) > 1:
                    time_diff = time.time() - self._start_time  # 计算总运行时间
                    samples_processed = len(self.metrics['loss'])  # 获取已处理的样本数
                    throughput = samples_processed / time_diff  # 计算吞吐量
                    self._history['throughput'].append(throughput)  # 记录吞吐量
                    self.metrics['throughput'].append(throughput)  # 记录到指标字典
                    
                # 根据interval调整睡眠时间
                elapsed = time.time() - start_time  # 计算本次循环耗时
                sleep_time = max(0, self.interval - elapsed)  # 计算剩余睡眠时间
                time.sleep(sleep_time)  # 睡眠等待
                
        except Exception as e:
            logger.error(f'Monitoring failed: {str(e)}', exc_info=True)  # 记录错误日志
            self._monitoring = False  # 标记监控已停止
            
    def log_system_info(self):
        """记录系统详细信息到日志"""
        try:
            logger.info('System Information:')
            logger.info(f'- Platform: {platform.platform()}')  # 操作系统平台
            logger.info(f'- Processor: {platform.processor()}')  # CPU处理器信息
            logger.info(f'- Python: {platform.python_version()}')  # Python版本
            logger.info(f'- PyTorch: {torch.__version__}')  # PyTorch版本
            
            if torch.cuda.is_available():
                logger.info('CUDA Information:')
                for i in range(torch.cuda.device_count()):
                    logger.info(f'  Device {i}: {torch.cuda.get_device_name(i)}')  # GPU设备名称
                    logger.info(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB')  # GPU显存大小
                    logger.info(f'    Compute Capability: {torch.cuda.get_device_capability(i)}')  # 计算能力
            else:
                logger.info('CUDA not available')  # 无可用CUDA
                
            # 记录CPU信息
            cpu_info = psutil.cpu_freq()
            if cpu_info:
                logger.info(f'- CPU Frequency: {cpu_info.current:.2f} MHz')  # CPU频率
                logger.info(f'- CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical')  # CPU核心数
                
            # 记录内存信息
            mem_info = psutil.virtual_memory()
            logger.info(f'- Total Memory: {mem_info.total / (1024 ** 3):.2f} GB')  # 总内存
            logger.info(f'- Available Memory: {mem_info.available / (1024 ** 3):.2f} GB')  # 可用内存
            
        except Exception as e:
            logger.error(f'Failed to log system info: {str(e)}')  # 记录错误日志
            
    def generate_report(self) -> Dict[str, any]:
        """生成资源使用报告
        
        Returns:
            Dict[str, any]: 包含各项资源使用统计的报告
            
        Raises:
            RuntimeError: 如果监控未启动
        """
        try:
            if not self._start_time:
                raise RuntimeError('Monitoring not started')  # 检查监控是否已启动
                
            duration = time.time() - self._start_time  # 计算监控总时长
            cpu_avg = sum(self._history['cpu']) / len(self._history['cpu']) if self._history['cpu'] else 0  # 计算CPU平均使用率
            memory_avg = sum(self._history['memory']) / len(self._history['memory']) if self._history['memory'] else 0  # 计算内存平均使用率
            gpu_avg = sum(self._history['gpu']) / len(self._history['gpu']) if self._history['gpu'] else 0  # 计算GPU平均使用率
            throughput_avg = sum(self._history['throughput']) / len(self._history['throughput']) if self._history['throughput'] else 0  # 计算平均吞吐量
            
            return {
                'duration': duration,  # 监控总时长
                'cpu_usage': cpu_avg,  # CPU平均使用率
                'memory_usage': memory_avg,  # 内存平均使用率
                'gpu_usage': gpu_avg,  # GPU平均使用率
                'throughput': throughput_avg,  # 平均吞吐量
                'metrics': self.metrics  # 所有监控指标
            }
            
        except Exception as e:
            logger.error(f'Failed to generate report: {str(e)}')  # 记录错误日志
            raise  # 重新抛出异常
