#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估脚本

该脚本用于评估模型性能
"""

import os
import sys
import argparse
import logging
import json
import yaml
from typing import Dict, Any, List
from pathlib import Path
import numpy as np

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.model.implementations.deepseek_model import DeepSeekModel
from app.utils.logging import setup_logging
from app.utils.metrics import calculate_perplexity, calculate_classification_metrics
from datasets import load_from_disk, Dataset

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

def evaluate_perplexity(model: DeepSeekModel, dataset: Dataset) -> float:
    """
    评估模型困惑度
    
    Args:
        model: 模型实例
        dataset: 评估数据集
        
    Returns:
        困惑度值
    """
    metrics = model.evaluate(dataset)
    loss = metrics.get("eval_loss", 0.0)
    perplexity = calculate_perplexity(loss)
    return perplexity

def evaluate_generation(model: DeepSeekModel, dataset: Dataset, max_samples: int = 100) -> Dict[str, Any]:
    """
    评估模型生成能力
    
    Args:
        model: 模型实例
        dataset: 评估数据集
        max_samples: 最大评估样本数
        
    Returns:
        评估结果
    """
    # 限制评估样本数
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    results = []
    
    for item in dataset:
        prompt = item["instruction"]
        reference = item["response"]
        
        # 生成文本
        generated_texts = model.generate(
            prompt=prompt,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1
        )
        
        generated_text = generated_texts[0] if generated_texts else ""
        
        # 保存结果
        results.append({
            "prompt": prompt,
            "reference": reference,
            "generated": generated_text
        })
    
    return {
        "samples": results,
        "num_samples": len(results)
    }

def save_results(results: Dict[str, Any], output_file: str) -> None:
    """
    保存评估结果到文件
    
    Args:
        results: 评估结果
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--data-dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--max-samples", type=int, default=100, help="最大评估样本数")
    parser.add_argument("--quantization", type=str, choices=["int8", "int4"], help="量化类型")
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
    logger.info("开始模型评估")
    
    # 加载配置
    config = {}
    if args.config:
        logger.info(f"加载配置: {args.config}")
        config = load_config(args.config)
    
    # 加载测试集
    test_path = os.path.join(args.data_dir, "test")
    if os.path.exists(test_path):
        logger.info(f"加载测试集: {test_path}")
        test_dataset = load_from_disk(test_path)
        logger.info(f"测试集大小: {len(test_dataset)}")
    else:
        # 如果没有测试集，尝试加载验证集
        val_path = os.path.join(args.data_dir, "validation")
        if os.path.exists(val_path):
            logger.info(f"未找到测试集，加载验证集: {val_path}")
            test_dataset = load_from_disk(val_path)
            logger.info(f"验证集大小: {len(test_dataset)}")
        else:
            logger.error("未找到测试集或验证集")
            sys.exit(1)
    
    # 创建模型
    logger.info(f"加载模型: {args.model_path}")
    model = DeepSeekModel(model_path=args.model_path)
    
    # 加载模型
    model.load_model(quantization=args.quantization)
    
    # 评估困惑度
    logger.info("评估模型困惑度")
    perplexity = evaluate_perplexity(model, test_dataset)
    logger.info(f"困惑度: {perplexity:.4f}")
    
    # 评估生成能力
    logger.info("评估模型生成能力")
    generation_results = evaluate_generation(
        model, 
        test_dataset, 
        max_samples=args.max_samples
    )
    logger.info(f"评估了 {generation_results['num_samples']} 个样本")
    
    # 汇总结果
    results = {
        "model_path": args.model_path,
        "metrics": {
            "perplexity": perplexity
        },
        "generation_results": generation_results
    }
    
    # 保存结果
    if args.output:
        logger.info(f"保存评估结果到: {args.output}")
        save_results(results, args.output)
    
    logger.info("评估完成")

if __name__ == "__main__":
    main() 