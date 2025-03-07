import os
import torch
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_BASE_PATH
from llm_finetuning_project.infrastructure.repositories.model_repository import ModelRepository

logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.scaler = None
        self._monitoring = False
        self._monitor_thread = None

    def train(self, config, training_data_path, checkpoint_path=None):
        """Train model with given configuration using mixed precision"""
        try:
            training_record = ModelRepository.create_training_record(config)
            
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
                'loss': [],
                'learning_rate': [],
                'epoch_times': [],
                'memory_usage': [],
                'gpu_memory': [],
                'throughput': [],
                'cpu_usage': []
            }
            
            self._start_resource_monitoring(metrics)
            
            if config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                logger.info('Enabled gradient checkpointing')
            
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
            if config.use_amp:
                logger.info('Enabled mixed precision training')
            
            start_epoch = 0
            if checkpoint_path:
                logger.info(f'Loading checkpoint from {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f'Resuming training from epoch {start_epoch}')
            
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs
            )
            
            for epoch in range(start_epoch, config.num_epochs):
                self._train_epoch(epoch, config, optimizer, scheduler, metrics)
                
                if (epoch + 1) % config.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(
                        MODEL_BASE_PATH,
                        f'checkpoint_epoch_{epoch + 1}'
                    )
                    self._save_checkpoint(checkpoint_path, epoch, optimizer, scheduler)
            
            ModelRepository.update_training_record(training_record.id, 'completed')
            self._stop_resource_monitoring()
            self._cleanup_resources()
            
            return training_record.id
            
        except Exception as e:
            logger.error(f'Training failed: {str(e)}', exc_info=True)
            self._handle_training_failure(e, epoch, checkpoint_path, training_record)
            raise

    def _train_epoch(self, epoch, config, optimizer, scheduler, metrics):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        accumulation_steps = config.gradient_accumulation_steps
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss / accumulation_steps
            
            try:
                self.scaler.scale(loss).backward()
            except RuntimeError as e:
                self._handle_gradient_error(e, optimizer)
            
            if (step + 1) % accumulation_steps == 0:
                self._optimizer_step(config, optimizer, scheduler)
                total_loss += loss.detach().item() * accumulation_steps
                
                if step % 50 == 0:
                    self._log_gradient_stats(optimizer)
                    self._manage_memory()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch + 1}/{config.num_epochs} - Loss: {avg_loss:.4f}')
        metrics['loss'].append(avg_loss)
        metrics['learning_rate'].append(scheduler.get_last_lr()[0])
