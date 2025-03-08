#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型推理脚本

该脚本用于使用训练好的模型进行文本生成
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.model.implementations.deepseek_model import DeepSeekModel
from app.utils.logging import setup_logging

def load_prompts(file_path: str) -> List[str]:
    """
    从文件加载提示文本
    
    Args:
        file_path: 文件路径
        
    Returns:
        提示文本列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    return data
                elif all(isinstance(item, dict) and 'prompt' in item for item in data):
                    return [item['prompt'] for item in data]
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
            raise ValueError("JSON文件格式不正确，无法提取提示文本")
        else:
            return [line.strip() for line in f if line.strip()]

def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    保存生成结果到文件
    
    Args:
        results: 生成结果列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--prompts", type=str, help="提示文本文件路径")
    parser.add_argument("--prompt", type=str, help="单个提示文本")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--max-length", type=int, default=512, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top-p", type=float, default=0.9, help="核采样参数")
    parser.add_argument("--top-k", type=int, default=50, help="top-k采样参数")
    parser.add_argument("--num-return-sequences", type=int, default=1, help="返回序列数量")
    parser.add_argument("--quantization", type=str, choices=["int8", "int4"], help="量化类型")
    parser.add_argument("--log-file", type=str, help="日志文件路径")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别")
    
    args = parser.parse_args()
    
    # 检查提示文本
    if not args.prompt and not args.prompts:
        parser.error("必须提供--prompt或--prompts参数")
    
    # 设置日志
    setup_logging(
        log_file=args.log_file,
        log_level=getattr(logging, args.log_level)
    )
    
    logger = logging.getLogger(__name__)
    logger.info("开始模型推理")
    
    # 加载提示文本
    if args.prompts:
        logger.info(f"从文件加载提示文本: {args.prompts}")
        prompts = load_prompts(args.prompts)
        logger.info(f"加载了 {len(prompts)} 条提示文本")
    else:
        prompts = [args.prompt]
        logger.info(f"使用单个提示文本: {args.prompt}")
    
    # 创建模型
    logger.info(f"加载模型: {args.model_path}")
    model = DeepSeekModel(model_path=args.model_path)
    
    # 加载模型
    model.load_model(quantization=args.quantization)
    
    # 生成结果
    results = []
    for i, prompt in enumerate(prompts):
        logger.info(f"处理提示文本 {i+1}/{len(prompts)}")
        
        generated_texts = model.generate(
            prompt=prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences
        )
        
        result = {
            "prompt": prompt,
            "generated_texts": generated_texts
        }
        results.append(result)
        
        # 打印结果
        logger.info(f"提示: {prompt}")
        for j, text in enumerate(generated_texts):
            logger.info(f"生成 {j+1}: {text}")
    
    # 保存结果
    if args.output:
        logger.info(f"保存结果到: {args.output}")
        save_results(results, args.output)
    
    logger.info("推理完成")

if __name__ == "__main__":
    main() 