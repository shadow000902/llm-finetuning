"""
模型服务模块
实现模型训练、评估、预测等核心功能
包含数据预处理、资源监控、模型评估等组件
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from torch import nn
from torch.utils.data import DataLoader

class TrainingError(Exception):
    """
    训练异常类
    
    在模型训练过程中发生错误时抛出，包括但不限于以下场景：
    - 数据加载失败
    - 模型参数更新失败
    - 资源不足导致训练中断
    - 早停条件触发
    - 验证集评估失败
    
    继承自Exception类，可通过try-except捕获处理
    """
    pass
from app.core.interfaces.model_operations import IModelOperations
from app.core.services.model_components import (
    DataProcessor,
    ModelTrainer,
    ResourceMonitor,
    ModelEvaluator,
    ModelInference
)

logger = logging.getLogger(__name__)

class ModelService(IModelOperations):
    """
    模型服务类
    实现IModelOperations接口，提供模型训练、评估、预测等核心功能
    包含以下组件：
    - 数据处理器：负责数据预处理和加载
    - 模型训练器：负责模型训练过程
    - 资源监控器：监控训练过程中的资源使用情况
    - 模型评估器：评估模型性能
    - 模型推理器：处理模型推理任务
    """
    
    def __init__(self,
                model: torch.nn.Module,
                device: torch.device,
                num_classes: int,
                class_names: Optional[List[str]] = None):
        """
        初始化模型服务
        
        该方法初始化模型服务实例，设置核心参数并初始化各功能组件
        
        Args:
            model: PyTorch模型实例，需要实现forward方法
            device: 计算设备 (CPU/GPU)，用于指定模型和数据所在的设备
            num_classes: 分类任务中的类别数量，用于评估器初始化
            class_names: 可选的类别名称列表，用于生成可读性更好的评估报告
            
        初始化过程：
        1. 设置模型、设备、类别数量等核心参数
        2. 初始化各功能组件：
           - 数据处理器：负责数据预处理和加载
           - 模型训练器：负责模型训练过程
           - 资源监控器：监控训练过程中的资源使用情况
           - 模型评估器：评估模型性能
           - 模型推理器：处理模型推理任务
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names
        
        # 初始化各功能组件
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer(model, device)
        self.resource_monitor = ResourceMonitor(self.model_trainer.metrics)
        self.model_evaluator = ModelEvaluator(num_classes, class_names)
        self.model_inference = ModelInference(model, device)
        
    def train(self,
             train_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
             val_data: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
             epochs: int = 10,
             batch_size: int = 32,
             learning_rate: float = 0.001,
             early_stopping_patience: int = 5) -> Dict[str, Any]:
        """
        训练模型并监控训练过程
        
        该方法执行完整的模型训练流程，包括：
        - 数据加载器准备
        - 多轮次训练
        - 验证集评估
        - 资源监控
        - 早停机制
        - 训练报告生成
        
        训练流程详细说明：
        1. 启动资源监控器
        2. 准备训练和验证数据加载器
        3. 执行多轮次训练：
           - 每个epoch执行一次完整训练
           - 在验证集上评估模型性能
           - 检查早停条件
        4. 生成训练报告
        5. 停止资源监控器
        
        Args:
            train_data: 训练数据，可以是以下形式：
                        - 单个张量：适用于简单数据集
                        - 字典形式：适用于复杂数据集，键为特征名，值为对应张量
            val_data: 可选验证数据，格式与train_data相同，用于评估模型性能
            epochs: 训练轮次，默认10
            batch_size: 批量大小，默认32
            learning_rate: 学习率，默认0.001
            early_stopping_patience: 早停等待轮次，默认5。当验证集性能连续patience轮次
                                   没有提升时，提前停止训练
                                    
        Returns:
            Dict[str, Any]: 包含训练、评估和资源使用情况的报告字典，结构如下：
            {
                'training': {
                    'epochs_completed': 实际完成的训练轮次,
                    'total_loss': 总损失值,
                    'learning_rate': 最终学习率,
                    'batch_size': 使用的批量大小
                },
                'evaluation': {
                    'accuracy': 准确率,
                    'precision': 精确率,
                    'recall': 召回率,
                    'f1_score': F1分数,
                    'confusion_matrix': 混淆矩阵
                },
                'resources': {
                    'cpu_usage': CPU使用率统计,
                    'memory_usage': 内存使用统计,
                    'gpu_usage': GPU使用统计（如果可用）
                }
            }
            
        Raises:
            TrainingError: 当训练过程中发生以下错误时抛出：
                           - 数据加载失败
                           - 模型参数更新失败
                           - 资源不足导致训练中断
                           - 验证集评估失败
        """
        self.resource_monitor.start()
        
        try:
            train_loader, val_loader = self._prepare_data_loaders(
                train_data, val_data, batch_size)
            
            for epoch in range(epochs):
                self._train_epoch(epoch, epochs, train_loader, val_loader)
                
                if self._should_stop_early(early_stopping_patience):
                    break
                    
            return self._generate_reports()
            
        except Exception as e:
            logger.error(f'Training failed: {str(e)}')
            raise TrainingError(f'Training failed: {str(e)}') from e
        finally:
            self.resource_monitor.stop()

    def _prepare_data_loaders(
        self,
        train_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        val_data: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        batch_size: int
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare data loaders for training"""
        train_loader = self.data_processor.create_dataloader(train_data, batch_size)
        val_loader = None
        if val_data is not None:
            val_loader = self.data_processor.create_dataloader(val_data, batch_size)
        return train_loader, val_loader

    def _train_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader]
    ) -> None:
        """Train for one epoch"""
        logger.info(f'Starting epoch {epoch + 1}/{total_epochs}')
        
        train_metrics = self.model_trainer.train_epoch(train_loader)
        
        if val_loader is not None:
            val_metrics = self.model_trainer.validate(val_loader)
            self._update_evaluation_metrics(val_metrics)

    def _update_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update evaluation metrics with validation results"""
        self.model_evaluator.update_batch_metrics(
            metrics['predictions'],
            metrics['targets'],
            metrics['loss']
        )

    def _should_stop_early(self, patience: int) -> bool:
        """Check if early stopping should be triggered"""
        if self.model_evaluator.early_stopping_check(patience):
            logger.info('Early stopping triggered')
            return True
        return False

    def _generate_reports(self) -> Dict[str, Any]:
        """Generate training, evaluation and resource reports"""
        return {
            'training': self.model_trainer.generate_training_report(),
            'evaluation': self.model_evaluator.generate_evaluation_report(),
            'resources': self.resource_monitor.generate_report()
        }
            
    def evaluate(self,
                data: Union[torch.Tensor, Dict[str, torch.Tensor]],
                batch_size: int = 32) -> Dict[str, Any]:
        """
        评估模型性能
        
        该方法在给定数据集上评估模型性能，包括：
        - 创建数据加载器
        - 计算模型指标
        - 更新评估器状态
        - 生成评估报告
        
        评估流程详细说明：
        1. 准备数据加载器
        2. 在数据集上执行模型推理
        3. 计算以下指标：
           - 准确率：正确预测的样本比例
           - 精确率：预测为正类的样本中实际为正类的比例
           - 召回率：实际为正类的样本中被正确预测的比例
           - F1分数：精确率和召回率的调和平均数
           - 混淆矩阵：展示预测结果与实际结果的对比矩阵
        4. 更新评估器内部状态
        5. 生成评估报告
        
        Args:
            data: 评估数据，可以是以下形式：
                  - 单个张量：适用于简单数据集
                  - 字典形式：适用于复杂数据集，键为特征名，值为对应张量
            batch_size: 批量大小，默认32。较大的批量大小可以提高评估速度，
                       但需要更多内存
                        
        Returns:
            Dict[str, Any]: 包含模型评估结果的报告字典，结构如下：
            {
                'accuracy': 准确率（0-1之间的小数）,
                'precision': 精确率（0-1之间的小数）,
                'recall': 召回率（0-1之间的小数）,
                'f1_score': F1分数（0-1之间的小数）,
                'confusion_matrix': 混淆矩阵（numpy数组，形状为[num_classes, num_classes]）
            }
            
        Raises:
            ValueError: 当输入数据格式不正确时抛出
            RuntimeError: 当评估过程中发生错误时抛出
        """
        dataloader = self.data_processor.create_dataloader(data, batch_size)
        metrics = self.model_trainer.validate(dataloader)
        
        self.model_evaluator.update_batch_metrics(
            metrics['predictions'],
            metrics['targets'],
            metrics['loss']
        )
        
        return self.model_evaluator.generate_evaluation_report()
        
    def predict(self,
               inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
               batch_size: int = 32,
               return_probs: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        使用模型进行预测
        
        该方法对输入数据进行预测，可选择返回预测概率。支持批量预测以提高效率。
        
        预测流程详细说明：
        1. 准备数据加载器
        2. 执行模型推理：
           - 将输入数据转换为模型可接受的格式
           - 在指定设备上执行前向传播
           - 获取模型输出
        3. 处理模型输出：
           - 如果return_probs为False，返回预测类别
           - 如果return_probs为True，返回预测类别和对应概率
        4. 将结果转换为numpy数组返回
        
        Args:
            inputs: 输入数据，可以是以下形式：
                   - 单个张量：适用于简单输入
                   - 字典形式：适用于复杂输入，键为特征名，值为对应张量
            batch_size: 批量大小，默认32。较大的批量大小可以提高推理速度，
                       但需要更多内存
            return_probs: 是否返回预测概率，默认False。当需要了解模型预测的
                          置信度时设置为True
                           
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            - 如果return_probs为False，返回预测结果数组，形状为[num_samples]
            - 如果return_probs为True，返回元组：
              (预测结果数组, 预测概率数组)，其中：
              - 预测结果数组形状为[num_samples]
              - 预测概率数组形状为[num_samples, num_classes]
              
        Raises:
            ValueError: 当输入数据格式不正确时抛出
            RuntimeError: 当推理过程中发生错误时抛出
        """
        return self.model_inference.predict(inputs, batch_size, return_probs)
        
    def save_model(self, path: str) -> None:
        """
        保存模型状态到指定路径
        
        该方法将当前模型的状态字典保存到指定路径，可用于模型持久化
        
        Args:
            path: 模型保存路径，应包含文件名和扩展名（通常为.pt或.pth）
            
        Raises:
            IOError: 当文件保存失败时抛出
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f'Saved model to {path}')
        
    def load_model(self, path: str) -> nn.Module:
        """
        从指定路径加载模型状态
        
        该方法从指定路径加载模型的状态字典，恢复模型到保存时的状态
        
        Args:
            path: 模型状态文件路径，应包含文件名和扩展名（通常为.pt或.pth）
            
        Returns:
            nn.Module: 加载后的模型实例
            
        Raises:
            IOError: 当文件加载失败时抛出
            RuntimeError: 当模型结构与状态字典不匹配时抛出
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f'Loaded model from {path}')

    def generate_text(self,
                     prompt: str,
                     max_length: int = 50,
                     temperature: float = 0.7) -> str:
        """
        根据提示文本生成新文本
        
        该方法使用模型根据给定的提示文本生成新的文本内容，支持控制生成长度和随机性
        
        Args:
            prompt: 提示文本，作为生成过程的起始点
            max_length: 生成文本的最大长度，默认50
            temperature: 控制生成随机性的温度参数，值越大生成结果越随机，默认0.7
            
        Returns:
            str: 生成的文本内容
            
        Raises:
            ValueError: 当提示文本为空或无效时抛出
            RuntimeError: 当文本生成失败时抛出
        """
        try:
            # Tokenize input prompt
            input_ids = self.data_processor.tokenize_text(prompt)
            
            # Generate text
            output_ids = self.model_inference.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature
            )
            
            # Decode generated text
            generated_text = self.data_processor.decode_text(output_ids)
            return generated_text
            
        except Exception as e:
            logger.error(f'Text generation failed: {str(e)}')
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型基本信息
        
        返回包含模型类型、类别数量、类别名称和设备信息的字典
        
        Returns:
            Dict[str, Any]: 包含模型信息的字典，结构如下：
            {
                'model_type': 模型类型名称,
                'num_classes': 类别数量,
                'class_names': 类别名称列表,
                'device': 计算设备信息
            }
        """
        return {
            'model_type': type(self.model).__name__,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'device': str(self.device)
        }
