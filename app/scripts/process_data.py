#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理脚本

该脚本用于将原始数据转换为模型训练所需的格式
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datasets import Dataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    加载JSON格式的数据文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        数据列表
    """
    logger.info(f"加载JSON数据: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载 {len(data)} 条数据")
        return data
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    加载CSV格式的数据文件
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        数据DataFrame
    """
    logger.info(f"加载CSV数据: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"成功加载 {len(df)} 条数据")
        return df
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise

def format_instruction_data(data: List[Dict[str, Any]], template: str = None) -> List[Dict[str, str]]:
    """
    将指令数据格式化为模型训练所需的格式
    
    Args:
        data: 原始数据列表
        template: 提示模板，默认为None
        
    Returns:
        格式化后的数据列表
    """
    logger.info("格式化指令数据")
    formatted_data = []
    
    # 默认模板
    default_template = """### 指令:
{instruction}

### 回答:
{response}"""
    
    template = template or default_template
    
    for item in data:
        if "instruction" in item and "response" in item:
            # 使用模板格式化数据
            text = template.format(
                instruction=item["instruction"],
                response=item["response"]
            )
            
            formatted_data.append({
                "text": text,
                "instruction": item["instruction"],
                "response": item["response"]
            })
    
    logger.info(f"格式化完成，共 {len(formatted_data)} 条数据")
    return formatted_data

def create_huggingface_dataset(data: List[Dict[str, str]]) -> Dataset:
    """
    创建HuggingFace数据集
    
    Args:
        data: 格式化后的数据列表
        
    Returns:
        HuggingFace数据集
    """
    logger.info("创建HuggingFace数据集")
    return Dataset.from_pandas(pd.DataFrame(data))

def split_dataset(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Dict[str, Dataset]:
    """
    将数据集分割为训练集、验证集和测试集
    
    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        包含训练集、验证集和测试集的字典
    """
    logger.info(f"分割数据集: 训练集 {train_ratio}, 验证集 {val_ratio}, 测试集 {test_ratio}")
    
    # 检查比例和是否为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"比例总和不为1: {total_ratio}，将进行归一化")
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
    
    # 计算各部分大小
    train_size = train_ratio
    val_size = val_ratio
    test_size = test_ratio
    
    # 分割数据集
    splits = dataset.train_test_split(test_size=val_size + test_size, seed=seed)
    train_dataset = splits["train"]
    
    # 进一步分割验证集和测试集
    if test_size > 0:
        remaining_splits = splits["test"].train_test_split(
            test_size=test_size / (val_size + test_size), 
            seed=seed
        )
        val_dataset = remaining_splits["train"]
        test_dataset = remaining_splits["test"]
    else:
        val_dataset = splits["test"]
        test_dataset = Dataset.from_pandas(pd.DataFrame([]))
    
    logger.info(f"数据集分割完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")
    
    return {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }

def save_datasets(datasets: Dict[str, Dataset], output_dir: str) -> None:
    """
    保存数据集到指定目录
    
    Args:
        datasets: 包含训练集、验证集和测试集的字典
        output_dir: 输出目录
    """
    logger.info(f"保存数据集到: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存各部分数据集
    for split_name, dataset in datasets.items():
        if len(dataset) > 0:
            split_dir = os.path.join(output_dir, split_name)
            dataset.save_to_disk(split_dir)
            logger.info(f"已保存 {split_name} 数据集: {len(dataset)} 条")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据处理脚本")
    parser.add_argument("--input", type=str, required=True, help="输入数据文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--format", type=str, choices=["json", "csv"], default="json", help="输入数据格式")
    parser.add_argument("--template", type=str, help="提示模板")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.format == "json":
        data = load_json_data(args.input)
    else:
        df = load_csv_data(args.input)
        data = df.to_dict("records")
    
    # 格式化数据
    formatted_data = format_instruction_data(data, args.template)
    
    # 创建数据集
    dataset = create_huggingface_dataset(formatted_data)
    
    # 分割数据集
    split_datasets = split_dataset(
        dataset, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 保存数据集
    save_datasets(split_datasets, args.output)
    
    logger.info("数据处理完成")

if __name__ == "__main__":
    main() 