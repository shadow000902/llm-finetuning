"""
DeepSeek-R1 1.5B模型实现

该模块提供了DeepSeek-R1 1.5B模型的具体实现
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

from app.core.interfaces.model_operations import ModelOperationsInterface
from app.utils.logging import get_logger

logger = get_logger(__name__)

class DeepSeekModel(ModelOperationsInterface):
    """DeepSeek-R1 1.5B模型的实现类"""
    
    def __init__(self, model_path: str = "deepseek-ai/deepseek-llm-1.5b-base", device: str = None):
        """
        初始化DeepSeek模型
        
        Args:
            model_path: 模型路径或名称
            device: 设备类型，如'cuda:0'或'cpu'
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        logger.info(f"初始化DeepSeek模型，模型路径: {model_path}, 设备: {self.device}")
        
    def load_model(self, quantization: str = None) -> None:
        """
        加载模型和分词器
        
        Args:
            quantization: 量化类型，如'int8'或'int4'
        """
        logger.info(f"加载DeepSeek模型，量化类型: {quantization}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 设置模型加载参数
        model_kwargs = {
            "device_map": self.device,
            "trust_remote_code": True,
        }
        
        # 根据量化类型设置参数
        if quantization == "int8":
            model_kwargs["load_in_8bit"] = True
        elif quantization == "int4":
            model_kwargs["load_in_4bit"] = True
            model_kwargs["quantization_config"] = {
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        
        # 加载模型
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
            
    def prepare_for_training(self, lora_config: Dict[str, Any] = None) -> None:
        """
        准备模型进行训练（应用LoRA等）
        
        Args:
            lora_config: LoRA配置参数
        """
        if self.model is None:
            raise ValueError("模型尚未加载，请先调用load_model方法")
            
        logger.info("准备模型进行训练")
        
        # 默认LoRA配置
        default_lora_config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM
        }
        
        # 合并用户提供的配置
        if lora_config:
            default_lora_config.update(lora_config)
            
        # 准备模型进行量化训练
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            self.model = prepare_model_for_kbit_training(self.model)
            
        # 应用LoRA
        peft_config = LoraConfig(**default_lora_config)
        self.model = get_peft_model(self.model, peft_config)
        
        # 打印可训练参数信息
        self.model.print_trainable_parameters()
        logger.info("模型已准备好进行训练")
        
    def train(
        self, 
        train_dataset: Dataset, 
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[TrainingArguments] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            training_args: 训练参数
            **kwargs: 其他参数
            
        Returns:
            训练结果指标
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型或分词器尚未加载")
            
        logger.info("开始训练模型")
        
        # 默认训练参数
        default_args = TrainingArguments(
            output_dir="./models/checkpoints",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="tensorboard",
        )
        
        # 使用用户提供的训练参数覆盖默认参数
        args = training_args or default_args
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        train_result = trainer.train(**kwargs)
        metrics = train_result.metrics
        
        # 保存模型和分词器
        trainer.save_model()
        self.tokenizer.save_pretrained(args.output_dir)
        
        # 保存训练指标
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info(f"模型训练完成，指标: {metrics}")
        return metrics
        
    def generate(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        生成文本
        
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: 核采样参数
            top_k: top-k采样参数
            num_return_sequences: 返回序列数量
            **kwargs: 其他生成参数
            
        Returns:
            生成的文本列表
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型或分词器尚未加载")
            
        logger.info(f"生成文本，提示: {prompt[:50]}...")
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成参数
        generation_config = {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
            
        # 解码输出
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        logger.info(f"文本生成完成，生成了 {len(generated_texts)} 个序列")
        return generated_texts
        
    def save_model(self, output_dir: str) -> None:
        """
        保存模型和分词器
        
        Args:
            output_dir: 输出目录
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型或分词器尚未加载")
            
        logger.info(f"保存模型到 {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型和分词器
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("模型保存完成")
        
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            eval_dataset: 评估数据集
            
        Returns:
            评估指标
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型或分词器尚未加载")
            
        logger.info("开始评估模型")
        
        # 创建评估器
        eval_args = TrainingArguments(
            output_dir="./models/eval",
            per_device_eval_batch_size=8,
            report_to="none",
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建评估器
        evaluator = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 评估模型
        metrics = evaluator.evaluate()
        
        logger.info(f"模型评估完成，指标: {metrics}")
        return metrics 