#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型训练脚本

该脚本用于启动模型训练
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from datasets import Dataset
from transformers import TrainingArguments

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.model.implementations.deepseek_model import DeepSeekModel
from app.utils.logging import setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--data-dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--log-file", type=str, help="日志文件路径")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(
        log_file=args.log_file,
        log_level=getattr(logging, args.log_level)
    )
    
    logger = logging.getLogger(__name__)
    logger.info("开始模型训练")
    
    # 加载配置
    config = load_config(args.config)
    logger.info(f"加载配置: {config}")
    
    # 设置输出目录
    output_dir = args.output_dir or config.get("output_dir", "./models/checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预处理后的数据集
    logger.info(f"加载预处理数据集: {args.data_dir}")
    train_dataset = Dataset.load_from_disk(os.path.join(args.data_dir, "train"))
    val_dataset = Dataset.load_from_disk(os.path.join(args.data_dir, "validation"))
    logger.info(f"加载验证集: {len(val_dataset)} 条数据")
    
    # 创建模型
    model_path = config.get("model", {}).get("base_model", "deepseek-ai/deepseek-llm-1.5b-base")
    quantization = config.get("model", {}).get("quantization", None)
    
    logger.info(f"初始化模型: {model_path}, 量化: {quantization}")
    model = DeepSeekModel(model_path=model_path)
    
    # 加载模型
    model.load_model(quantization=quantization)
    
    # 准备LoRA训练
    lora_config = config.get("lora", {})
    logger.info(f"准备LoRA训练: {lora_config}")
    model.prepare_for_training(config={"lora": lora_config})
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("training", {}).get("batch_size", 8),
        gradient_accumulation_steps=config.get("training", {}).get("gradient_accumulation_steps", 1),
        num_train_epochs=config.get("training", {}).get("num_epochs", 3),
        learning_rate=config.get("training", {}).get("learning_rate", 3e-4),
        fp16=config.get("training", {}).get("fp16", True),
        logging_steps=config.get("training", {}).get("logging_steps", 100),
        save_steps=config.get("training", {}).get("save_steps", 1000),
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=config.get("training", {}).get("eval_steps", 500) if val_dataset else None,
        save_total_limit=config.get("training", {}).get("save_total_limit", 3),
        load_best_model_at_end=config.get("training", {}).get("load_best_model_at_end", True) if val_dataset else False,
        report_to=config.get("training", {}).get("report_to", "tensorboard"),
        remove_unused_columns=False
    )
    
    # 开始训练
    logger.info("开始训练模型")
    metrics = model.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_args=training_args
    )
    
    logger.info(f"训练完成，指标: {metrics}")
    
    # 保存模型
    final_output_dir = os.path.join(output_dir, "final")
    logger.info(f"保存最终模型到: {final_output_dir}")
    model.save_model(final_output_dir)
    
    logger.info("模型训练和保存完成")

if __name__ == "__main__":
    main()
