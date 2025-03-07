# 导入必要的库
import os  # 操作系统接口
import time  # 时间相关操作
import logging  # 日志记录
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
from typing import Dict, List, Optional, Union  # 类型注解
from collections import defaultdict  # 默认字典

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 模型推理类，封装了模型推理相关操作
class ModelInference:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """初始化模型推理器
        
        Args:
            model (torch.nn.Module): 要使用的模型
            device (torch.device): 模型运行的设备（CPU/GPU）
        """
        self.model = model  # 存储模型
        self.device = device  # 存储设备
        self.model.to(self.device)  # 将模型移动到指定设备
        self.model.eval()  # 设置模型为评估模式
        
    def predict(self,
               inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
               batch_size: int = 32,
               return_probs: bool = False) -> np.ndarray:
        """生成输入数据的预测结果
        
        Args:
            inputs: 输入数据，可以是张量或字典形式的张量
            batch_size: 批量大小，默认为32
            return_probs: 是否返回概率值，默认为False
            
        Returns:
            np.ndarray: 预测结果数组
        """
        # 处理输入数据
        if isinstance(inputs, dict):
            dataset = inputs  # 如果输入是字典，直接使用
        else:
            dataset = torch.utils.data.TensorDataset(inputs)  # 否则创建TensorDataset
            
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False  # 推理时不打乱数据
        )
        
        predictions = []  # 存储预测结果
        with torch.no_grad():  # 禁用梯度计算
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # 获取实际数据
                    
                batch = batch.to(self.device)  # 将数据移动到设备
                outputs = self.model(batch)  # 模型推理
                
                # 处理输出结果
                if return_probs:
                    if outputs.dim() > 1:
                        outputs = torch.softmax(outputs, dim=1)  # 多分类使用softmax
                    else:
                        outputs = torch.sigmoid(outputs)  # 二分类使用sigmoid
                        
                predictions.append(outputs.cpu())  # 将结果移回CPU
                
        return torch.cat(predictions).numpy()  # 合并所有batch的结果并转为numpy数组
        
    def predict_classes(self,
                      inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                      batch_size: int = 32) -> np.ndarray:
        """生成类别预测结果
        
        Args:
            inputs: 输入数据，可以是张量或字典形式的张量
            batch_size: 批量大小，默认为32
            
        Returns:
            np.ndarray: 类别预测结果数组
        """
        predictions = self.predict(inputs, batch_size)  # 获取预测结果
        return np.argmax(predictions, axis=1)  # 返回概率最大的类别索引
        
    def predict_probs(self,
                    inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                    batch_size: int = 32) -> np.ndarray:
        """生成概率预测结果
        
        Args:
            inputs: 输入数据，可以是张量或字典形式的张量
            batch_size: 批量大小，默认为32
            
        Returns:
            np.ndarray: 概率预测结果数组
        """
        return self.predict(inputs, batch_size, return_probs=True)  # 返回概率值
        
    def predict_with_threshold(self,
                             inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                             threshold: float = 0.5,
                             batch_size: int = 32) -> np.ndarray:
        """使用阈值生成二分类预测结果
        
        Args:
            inputs: 输入数据，可以是张量或字典形式的张量
            threshold: 分类阈值，默认为0.5
            batch_size: 批量大小，默认为32
            
        Returns:
            np.ndarray: 二分类预测结果数组（0或1）
        """
        probs = self.predict_probs(inputs, batch_size)  # 获取概率值
        return (probs > threshold).astype(int)  # 根据阈值进行二分类
        
    def generate_prediction_report(self,
                                 inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                 targets: Optional[np.ndarray] = None,
                                 batch_size: int = 32) -> Dict[str, any]:
        """生成全面的预测报告
        
        Args:
            inputs: 输入数据，可以是张量或字典形式的张量
            targets: 真实标签，可选
            batch_size: 批量大小，默认为32
            
        Returns:
            Dict[str, any]: 包含各种预测指标的字典
            
        Raises:
            ValueError: 当targets长度与predictions不匹配时抛出
        """
        predictions = self.predict_classes(inputs, batch_size)  # 获取类别预测
        probs = self.predict_probs(inputs, batch_size)  # 获取概率值
        
        # 构建基础报告
        report = {
            'predictions': predictions,  # 预测类别
            'probabilities': probs,  # 预测概率
            'input_shape': inputs.shape,  # 输入数据形状
            'num_samples': len(inputs)  # 样本数量
        }
        
        # 如果有真实标签，计算准确率等指标
        if targets is not None:
            if len(targets) != len(predictions):
                raise ValueError('Targets length does not match predictions')
                
            report['accuracy'] = np.mean(predictions == targets)  # 准确率
            report['correct_predictions'] = np.sum(predictions == targets)  # 正确预测数
            report['incorrect_predictions'] = np.sum(predictions != targets)  # 错误预测数
            
        return report
        
    def save_predictions(self,
                       inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                       output_path: str,
                       batch_size: int = 32) -> None:
        """保存预测结果到文件
        
        Args:
            inputs: 输入数据，可以是张量或字典形式的张量
            output_path: 输出文件路径
            batch_size: 批量大小，默认为32
        """
        predictions = self.predict_classes(inputs, batch_size)  # 获取预测结果
        np.save(output_path, predictions)  # 保存为numpy文件
        logger.info(f'Saved predictions to {output_path}')  # 记录日志

    def generate(self,
                input_ids: torch.Tensor,
                max_length: int = 100,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.95,
                repetition_penalty: float = 1.2,
                num_return_sequences: int = 1) -> List[str]:
        """使用模型生成文本序列
        
        Args:
            input_ids: 输入token id张量
            max_length: 生成文本的最大长度
            temperature: 采样温度，控制生成多样性
            top_k: top-k采样参数
            top_p: top-p采样参数
            repetition_penalty: 重复惩罚系数
            num_return_sequences: 返回的序列数量
            
        Returns:
            List[str]: 生成的文本序列列表
            
        Raises:
            ValueError: 当输入参数不合法时抛出
            Exception: 当文本生成失败时抛出
        """
        try:
            # 参数校验
            if not isinstance(input_ids, torch.Tensor):
                raise ValueError('Input must be a torch.Tensor')
                
            if input_ids.dim() != 2:
                raise ValueError('Input tensor must be 2-dimensional')
                
            if max_length <= 0:
                raise ValueError('max_length must be positive')
                
            if temperature <= 0:
                raise ValueError('temperature must be positive')
                
            if top_k <= 0:
                raise ValueError('top_k must be positive')
                
            if not (0 < top_p <= 1):
                raise ValueError('top_p must be between 0 and 1')
                
            if repetition_penalty <= 0:
                raise ValueError('repetition_penalty must be positive')
                
            if num_return_sequences <= 0:
                raise ValueError('num_return_sequences must be positive')
                
            # 记录生成参数
            logger.info(f'Generating text with params: max_length={max_length}, '
                      f'temperature={temperature}, top_k={top_k}, top_p={top_p}, '
                      f'repetition_penalty={repetition_penalty}, '
                      f'num_return_sequences={num_return_sequences}')
                      
            input_ids = input_ids.to(self.device)  # 将输入移动到设备
            
            # 执行文本生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,  # 启用采样
                    pad_token_id=self.model.config.pad_token_id,  # 填充token
                    eos_token_id=self.model.config.eos_token_id  # 结束token
                )
                
            return outputs.cpu()  # 将结果移回CPU
            
        except Exception as e:
            logger.error(f'Text generation failed: {str(e)}')  # 记录错误日志
            raise  # 重新抛出异常
