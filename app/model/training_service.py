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

    def train_model(self, config_path: str, task_id: str = None):
        """训练模型的主要方法
        
        Args:
            config_path (str): 训练配置文件路径
            task_id (str, optional): 训练任务ID，用于标识和追踪训练过程
            
        Returns:
            dict: 包含训练结果和指标的字典
        """
        try:
            # 加载训练配置
            config = self.load_config(config_path)
            logger.info(f"Loaded training configuration from {config_path}")
            
            # 初始化指标收集字典
            metrics = {
                'loss': [],
                'eval_loss': [],
                'learning_rate': [],
                'epoch_times': [],
                'cpu_usage': [],
                'memory_usage': [],
                'gpu_memory': [],
                'throughput': []
            }
            
            # 启动资源监控
            self._start_resource_monitoring(metrics)
            
            # 加载数据集
            train_dataset, eval_dataset = self.load_datasets(
                train_file=config['data']['train_file'],
                validation_file=config['data']['validation_file'],
                max_train_samples=config['data']['max_train_samples'],
                max_eval_samples=config['data']['max_eval_samples']
            )
            logger.info(f"Loaded training dataset with {len(train_dataset)} samples")
            logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples")
            
            # 加载预训练模型
            model, tokenizer = self.load_pretrained_model(
                model_name=config['model']['base_model'],
                max_length=config['model']['max_length']
            )
            logger.info(f"Loaded pretrained model: {config['model']['base_model']}")
            
            # 设置训练参数
            training_args = self.prepare_training_args(
                output_dir=config['model']['output_dir'],
                num_train_epochs=config['model']['num_epochs'],
                per_device_train_batch_size=config['model']['batch_size'],
                per_device_eval_batch_size=config['model']['batch_size'],
                learning_rate=config['model']['learning_rate'],
                weight_decay=config['model']['weight_decay'],
                warmup_steps=config['model']['warmup_steps'],
                logging_steps=config['model']['logging_steps'],
                save_steps=config['model']['save_steps'],
                fp16=config['training']['fp16'],
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
                gradient_checkpointing=config['training']['gradient_checkpointing'],
                evaluation_strategy=config['training']['evaluation_strategy'],
                eval_steps=config['training']['eval_steps'],
                save_total_limit=config['training']['save_total_limit'],
                load_best_model_at_end=config['training']['load_best_model_at_end'],
                metric_for_best_model=config['training']['metric_for_best_model'],
                greater_is_better=config['training']['greater_is_better'],
                report_to=config['logging']['report_to']
            )
            
            # 创建训练器
            trainer = self.create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_args=training_args,
                metrics=metrics
            )
            
            # 开始训练
            logger.info("Starting model training")
            start_time = time.time()
            train_result = trainer.train()
            end_time = time.time()
            
            # 记录训练时间
            training_time = end_time - start_time
            metrics['total_training_time'] = training_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # 评估模型
            eval_result = trainer.evaluate()
            metrics['final_eval_loss'] = eval_result['eval_loss']
            logger.info(f"Final evaluation loss: {eval_result['eval_loss']:.4f}")
            
            # 保存最终模型和tokenizer
            final_model_path = os.path.join(config['model']['output_dir'], 'final-model')
            trainer.save_model(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
            
            # 保存训练指标
            self.save_metrics(metrics, os.path.join(config['model']['output_dir'], 'training_metrics.json'))
            logger.info("Saved training metrics")
            
            # 停止资源监控
            self._stop_resource_monitoring()
            
            # 清理资源
            self._cleanup_resources()
            
            # 如果有任务ID，更新任务状态
            if task_id:
                model_dao = ModelDAO()
                model_dao.update_training_task(
                    task_id=task_id,
                    status="completed",
                    metrics=metrics,
                    model_path=final_model_path
                )
                logger.info(f"Updated training task {task_id} status to completed")
            
            return {
                'status': 'success',
                'model_path': final_model_path,
                'training_time': training_time,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            # 停止资源监控
            self._stop_resource_monitoring()
            # 清理资源
            self._cleanup_resources()
            # 如果有任务ID，更新任务状态为失败
            if task_id:
                model_dao = ModelDAO()
                model_dao.update_training_task(
                    task_id=task_id,
                    status="failed",
                    error=str(e)
                )
                logger.info(f"Updated training task {task_id} status to failed")
            
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def load_config(self, config_path: str) -> Dict:
        """加载YAML配置文件
        
        Args:
            config_path (str): 配置文件路径
            
        Returns:
            Dict: 配置字典
        """
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_datasets(self, train_file: str, validation_file: str, max_train_samples=None, max_eval_samples=None):
        """加载训练和评估数据集
        
        Args:
            train_file (str): 训练数据文件路径
            validation_file (str): 验证数据文件路径
            max_train_samples (int, optional): 最大训练样本数
            max_eval_samples (int, optional): 最大评估样本数
            
        Returns:
            tuple: (训练数据集, 评估数据集)
        """
        from datasets import load_dataset
        
        # 加载数据集
        data_files = {
            "train": train_file,
            "validation": validation_file
        }
        
        # 支持JSON和CSV格式
        extension = train_file.split(".")[-1]
        if extension == "csv":
            datasets = load_dataset("csv", data_files=data_files)
        elif extension == "json":
            datasets = load_dataset("json", data_files=data_files)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        # 限制样本数量
        if max_train_samples is not None:
            datasets["train"] = datasets["train"].select(range(min(len(datasets["train"]), max_train_samples)))
        
        if max_eval_samples is not None:
            datasets["validation"] = datasets["validation"].select(range(min(len(datasets["validation"]), max_eval_samples)))
        
        return datasets["train"], datasets["validation"]
    
    def load_pretrained_model(self, model_name: str, max_length: int):
        """加载预训练模型和分词器
        
        Args:
            model_name (str): 模型名称或路径
            max_length (int): 最大序列长度
            
        Returns:
            tuple: (模型, 分词器)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def prepare_training_args(self, **kwargs):
        """准备训练参数
        
        Args:
            **kwargs: 训练参数
            
        Returns:
            TrainingArguments: 训练参数对象
        """
        from transformers import TrainingArguments
        
        # 创建输出目录
        os.makedirs(kwargs.get('output_dir', './output'), exist_ok=True)
        
        # 创建训练参数
        training_args = TrainingArguments(**kwargs)
        
        return training_args
    
    def create_trainer(self, model, tokenizer, train_dataset, eval_dataset, training_args, metrics):
        """创建训练器
        
        Args:
            model: 模型
            tokenizer: 分词器
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            training_args: 训练参数
            metrics: 指标字典
            
        Returns:
            Trainer: 训练器对象
        """
        from transformers import Trainer, default_data_collator
        
        # 创建训练回调
        class MetricsCallback:
            def __init__(self, metrics_dict):
                self.metrics = metrics_dict
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    if 'loss' in logs:
                        self.metrics['loss'].append(logs['loss'])
                    if 'learning_rate' in logs:
                        self.metrics['learning_rate'].append(logs['learning_rate'])
                    if 'eval_loss' in logs:
                        self.metrics['eval_loss'].append(logs['eval_loss'])
                    
                    # 记录每个epoch的时间
                    if state.epoch > len(self.metrics['epoch_times']):
                        self.metrics['epoch_times'].append(state.epoch_time)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            callbacks=[MetricsCallback(metrics)]
        )
        
        return trainer
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """保存训练指标
        
        Args:
            metrics (Dict): 指标字典
            output_path (str): 输出文件路径
        """
        import json
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 将datetime对象转换为字符串
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], datetime):
                metrics[key] = [dt.isoformat() for dt in value]
        
        # 保存指标
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
