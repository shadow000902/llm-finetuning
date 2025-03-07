import os
import torch
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.config import MODEL_BASE_PATH
from app.repositories.model_repository import ModelRepository

logger = logging.getLogger(__name__)

class ModelTraining:
    """模型训练类，负责管理模型训练的全流程
    
    该类封装了模型训练的核心逻辑，包括：
    - 训练配置管理
    - 混合精度训练
    - 断点续训
    - 资源监控
    - 训练指标记录
    """
    def __init__(self, model, tokenizer, device):
        """初始化模型训练实例
        
        Args:
            model: 要训练的模型实例
            tokenizer: 文本分词器
            device: 训练设备（如'cuda'或'cpu'）
        """
        self.model = model  # 训练模型
        self.tokenizer = tokenizer  # 文本分词器
        self.device = device  # 训练设备
        self.scaler = None  # 混合精度训练的梯度缩放器
        self._monitoring = False  # 资源监控状态标志
        self._monitor_thread = None  # 资源监控线程

    def train(self, config, training_data_path, checkpoint_path=None):
        """使用给定配置训练模型，支持混合精度训练
        
        Args:
            config: 训练配置对象
            training_data_path: 训练数据路径
            checkpoint_path: 断点续训路径（可选）
            
        Returns:
            int: 训练记录ID
            
        Raises:
            Exception: 训练失败时抛出异常
        """
        try:
            training_record = ModelRepository.create_training_record(config)  # 创建训练记录
            
            logger.info('Starting training with configuration:')
            logger.info(f'- Model: {self.model.__class__.__name__}')
            logger.info(f'- Device: {self.device}')
            logger.info(f'- Batch size: {config.batch_size}')
            logger.info(f'- Learning rate: {config.learning_rate}')
            logger.info(f'- Epochs: {config.num_epochs}')
            logger.info(f'- Gradient accumulation steps: {config.gradient_accumulation_steps}')
            logger.info(f'- Mixed precision: {config.use_amp}')
            logger.info(f'- Gradient checkpointing: {config.gradient_checkpointing}')
            
            metrics = {
                'loss': [],  # 每个epoch的损失值
                'learning_rate': [],  # 每个epoch的学习率
                'epoch_times': [],  # 每个epoch的训练时间
                'memory_usage': [],  # 内存使用情况
                'gpu_memory': [],  # GPU显存使用情况
                'throughput': [],  # 训练吞吐量
                'cpu_usage': []  # CPU使用率
            }
            
            self._start_resource_monitoring(metrics)  # 启动资源监控
            
            if config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()  # 启用梯度检查点以节省显存
                logger.info('Enabled gradient checkpointing')
            
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)  # 初始化混合精度训练
            if config.use_amp:
                logger.info('Enabled mixed precision training')  # 启用混合精度训练
            
            start_epoch = 0
            if checkpoint_path:
                logger.info(f'Loading checkpoint from {checkpoint_path}')  # 加载断点
                checkpoint = torch.load(checkpoint_path)  # 加载检查点文件
                self.model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 加载调度器状态
                start_epoch = checkpoint['epoch'] + 1  # 设置起始epoch
                logger.info(f'Resuming training from epoch {start_epoch}')  # 记录续训信息
            
            optimizer = torch.optim.AdamW(  # 使用AdamW优化器
                self.model.parameters(),
                lr=config.learning_rate,  # 设置学习率
                weight_decay=config.weight_decay  # 设置权重衰减
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # 使用余弦退火学习率调度器
                optimizer,
                T_max=config.num_epochs  # 设置最大epoch数
            )
            
            for epoch in range(start_epoch, config.num_epochs):  # 遍历每个epoch
                self._train_epoch(epoch, config, optimizer, scheduler, metrics)  # 执行单个epoch训练
                
                if (epoch + 1) % config.checkpoint_interval == 0:  # 检查是否到达保存点
                    checkpoint_path = os.path.join(  # 构建检查点路径
                        MODEL_BASE_PATH,
                        f'checkpoint_epoch_{epoch + 1}'
                    )
                    self._save_checkpoint(checkpoint_path, epoch, optimizer, scheduler)  # 保存检查点
            
            ModelRepository.update_training_record(training_record.id, 'completed')  # 更新训练记录状态
            self._stop_resource_monitoring()  # 停止资源监控
            self._cleanup_resources()  # 清理资源
            
            return training_record.id  # 返回训练记录ID
            
        except Exception as e:
            logger.error(f'Training failed: {str(e)}', exc_info=True)  # 记录训练失败日志
            self._handle_training_failure(e, epoch, checkpoint_path, training_record)  # 处理训练失败
            raise  # 重新抛出异常

    def _train_epoch(self, epoch, config, optimizer, scheduler, metrics):
        """执行单个epoch的训练
        
        Args:
            epoch: 当前epoch序号
            config: 训练配置
            optimizer: 优化器
            scheduler: 学习率调度器
            metrics: 训练指标记录字典
            
        执行步骤：
        1. 设置模型为训练模式
        2. 初始化损失值
        3. 遍历训练数据
        4. 执行前向传播和反向传播
        5. 更新模型参数
        6. 记录训练指标
        """
        self.model.train()  # 设置模型为训练模式
        total_loss = 0  # 初始化总损失
        accumulation_steps = config.gradient_accumulation_steps  # 获取梯度累积步数
        optimizer.zero_grad()  # 清空梯度
        
        for step, batch in enumerate(train_loader):  # 遍历训练数据
            inputs = batch['input_ids'].to(self.device)  # 将输入数据移动到指定设备
            labels = batch['labels'].to(self.device)  # 将标签数据移动到指定设备
            
            with torch.cuda.amp.autocast(enabled=config.use_amp):  # 混合精度上下文
                outputs = self.model(inputs, labels=labels)  # 前向传播
                loss = outputs.loss / accumulation_steps  # 计算损失并考虑梯度累积
            
            try:
                self.scaler.scale(loss).backward()  # 反向传播（支持混合精度）
            except RuntimeError as e:
                self._handle_gradient_error(e, optimizer)  # 处理梯度错误
            
            if (step + 1) % accumulation_steps == 0:  # 检查是否达到梯度累积步数
                self._optimizer_step(config, optimizer, scheduler)  # 更新模型参数
                total_loss += loss.detach().item() * accumulation_steps  # 累加损失
                
                if step % 50 == 0:  # 每50步记录一次
                    self._log_gradient_stats(optimizer)  # 记录梯度统计信息
                    self._manage_memory()  # 管理内存使用
        
        avg_loss = total_loss / len(train_loader)  # 计算平均损失
        logger.info(f'Epoch {epoch + 1}/{config.num_epochs} - Loss: {avg_loss:.4f}')  # 记录epoch信息
        metrics['loss'].append(avg_loss)  # 记录损失指标
        metrics['learning_rate'].append(scheduler.get_last_lr()[0])  # 记录学习率
