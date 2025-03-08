"""
验证工具模块
提供请求参数验证和数据验证的工具函数
"""

from typing import Dict, List, Any, Optional, Union
import re

def validate_generation_request(data: Dict[str, Any]) -> List[str]:
    """
    验证生成请求
    
    Args:
        data: 生成请求数据
        
    Returns:
        错误消息列表，如果没有错误则为空列表
    """
    errors = []
    
    # 检查必需字段
    if 'prompt' not in data:
        errors.append("缺少必需字段 'prompt'")
    elif not isinstance(data['prompt'], str):
        errors.append("'prompt' 必须是一个字符串")
    elif len(data['prompt'].strip()) == 0:
        errors.append("'prompt' 不能为空")
    
    # 验证可选字段
    if 'max_length' in data:
        if not isinstance(data['max_length'], int):
            errors.append("'max_length' 必须是一个整数")
        elif data['max_length'] <= 0 or data['max_length'] > 2048:
            errors.append("'max_length' 必须在 1 到 2048 之间")
    
    if 'temperature' in data:
        if not isinstance(data['temperature'], (int, float)):
            errors.append("'temperature' 必须是一个数字")
        elif data['temperature'] < 0 or data['temperature'] > 2:
            errors.append("'temperature' 必须在 0 到 2 之间")
    
    if 'top_p' in data:
        if not isinstance(data['top_p'], (int, float)):
            errors.append("'top_p' 必须是一个数字")
        elif data['top_p'] <= 0 or data['top_p'] > 1:
            errors.append("'top_p' 必须在 0 到 1 之间")
    
    if 'top_k' in data:
        if not isinstance(data['top_k'], int):
            errors.append("'top_k' 必须是一个整数")
        elif data['top_k'] <= 0:
            errors.append("'top_k' 必须大于 0")
    
    if 'num_return_sequences' in data:
        if not isinstance(data['num_return_sequences'], int):
            errors.append("'num_return_sequences' 必须是一个整数")
        elif data['num_return_sequences'] <= 0 or data['num_return_sequences'] > 5:
            errors.append("'num_return_sequences' 必须在 1 到 5 之间")
    
    if 'model_path' in data and not isinstance(data['model_path'], str):
        errors.append("'model_path' 必须是一个字符串")
    
    return errors

def validate_training_config(data: Dict[str, Any]) -> List[str]:
    """
    验证训练配置
    
    Args:
        data: 训练配置数据
        
    Returns:
        错误消息列表，如果没有错误则为空列表
    """
    errors = []
    
    # 检查必需字段
    if 'dataset_path' not in data:
        errors.append("缺少必需字段 'dataset_path'")
    
    # 验证配置参数
    if 'config' in data:
        config = data['config']
        if not isinstance(config, dict):
            errors.append("'config' 必须是一个对象")
        else:
            # 验证学习率
            if 'learning_rate' in config and not isinstance(config['learning_rate'], (int, float)):
                errors.append("'learning_rate' 必须是一个数字")
            elif 'learning_rate' in config and (config['learning_rate'] <= 0 or config['learning_rate'] > 1):
                errors.append("'learning_rate' 必须在 0 到 1 之间")
                
            # 验证批次大小
            if 'batch_size' in config and not isinstance(config['batch_size'], int):
                errors.append("'batch_size' 必须是一个整数")
            elif 'batch_size' in config and config['batch_size'] <= 0:
                errors.append("'batch_size' 必须大于 0")
                
            # 验证训练轮数
            if 'num_epochs' in config and not isinstance(config['num_epochs'], int):
                errors.append("'num_epochs' 必须是一个整数")
            elif 'num_epochs' in config and config['num_epochs'] <= 0:
                errors.append("'num_epochs' 必须大于 0")
    
    # 验证LoRA配置
    if 'lora_config' in data:
        lora_config = data['lora_config']
        if not isinstance(lora_config, dict):
            errors.append("'lora_config' 必须是一个对象")
        else:
            # 验证LoRA秩
            if 'r' in lora_config and not isinstance(lora_config['r'], int):
                errors.append("'r' 必须是一个整数")
            elif 'r' in lora_config and lora_config['r'] <= 0:
                errors.append("'r' 必须大于 0")
                
            # 验证LoRA alpha
            if 'lora_alpha' in lora_config and not isinstance(lora_config['lora_alpha'], int):
                errors.append("'lora_alpha' 必须是一个整数")
            elif 'lora_alpha' in lora_config and lora_config['lora_alpha'] <= 0:
                errors.append("'lora_alpha' 必须大于 0")
                
            # 验证LoRA dropout
            if 'lora_dropout' in lora_config and not isinstance(lora_config['lora_dropout'], (int, float)):
                errors.append("'lora_dropout' 必须是一个数字")
            elif 'lora_dropout' in lora_config and (lora_config['lora_dropout'] < 0 or lora_config['lora_dropout'] > 1):
                errors.append("'lora_dropout' 必须在 0 到 1 之间")
    
    return errors

def validate_batch_generation_request(data: Dict[str, Any]) -> List[str]:
    """
    验证批量生成请求
    
    Args:
        data: 批量生成请求数据
        
    Returns:
        错误消息列表，如果没有错误则为空列表
    """
    errors = []
    
    # 检查必需字段
    if 'prompts' not in data:
        errors.append("缺少必需字段 'prompts'")
    elif not isinstance(data['prompts'], list):
        errors.append("'prompts' 必须是一个数组")
    elif len(data['prompts']) == 0:
        errors.append("'prompts' 不能为空")
    elif len(data['prompts']) > 50:
        errors.append("'prompts' 最多包含 50 个提示")
    else:
        # 验证每个提示
        for i, prompt in enumerate(data['prompts']):
            if not isinstance(prompt, str):
                errors.append(f"'prompts[{i}]' 必须是一个字符串")
            elif len(prompt.strip()) == 0:
                errors.append(f"'prompts[{i}]' 不能为空")
    
    # 验证可选字段
    if 'max_length' in data:
        if not isinstance(data['max_length'], int):
            errors.append("'max_length' 必须是一个整数")
        elif data['max_length'] <= 0 or data['max_length'] > 2048:
            errors.append("'max_length' 必须在 1 到 2048 之间")
    
    if 'temperature' in data:
        if not isinstance(data['temperature'], (int, float)):
            errors.append("'temperature' 必须是一个数字")
        elif data['temperature'] < 0 or data['temperature'] > 2:
            errors.append("'temperature' 必须在 0 到 2 之间")
    
    if 'top_p' in data:
        if not isinstance(data['top_p'], (int, float)):
            errors.append("'top_p' 必须是一个数字")
        elif data['top_p'] <= 0 or data['top_p'] > 1:
            errors.append("'top_p' 必须在 0 到 1 之间")
    
    if 'top_k' in data:
        if not isinstance(data['top_k'], int):
            errors.append("'top_k' 必须是一个整数")
        elif data['top_k'] <= 0:
            errors.append("'top_k' 必须大于 0")
    
    if 'num_return_sequences' in data:
        if not isinstance(data['num_return_sequences'], int):
            errors.append("'num_return_sequences' 必须是一个整数")
        elif data['num_return_sequences'] <= 0 or data['num_return_sequences'] > 5:
            errors.append("'num_return_sequences' 必须在 1 到 5 之间")
    
    if 'model_path' in data and not isinstance(data['model_path'], str):
        errors.append("'model_path' 必须是一个字符串")
    
    return errors

def validate_dataset_upload(request) -> List[str]:
    """
    验证数据集上传请求
    
    Args:
        request: Flask请求对象
        
    Returns:
        错误消息列表，如果没有错误则为空列表
    """
    errors = []
    
    # 检查文件
    if 'file' not in request.files:
        errors.append("缺少必需字段 'file'")
    else:
        file = request.files['file']
        if file.filename == '':
            errors.append("未选择文件")
        elif not file.filename.endswith(('.json', '.csv')):
            errors.append("文件格式必须是 JSON 或 CSV")
    
    # 检查名称
    if 'name' not in request.form:
        errors.append("缺少必需字段 'name'")
    elif not request.form['name'].strip():
        errors.append("'name' 不能为空")
    
    # 检查格式
    if 'format' in request.form:
        format = request.form['format']
        if format not in ('json', 'csv'):
            errors.append("'format' 必须是 'json' 或 'csv'")
    
    return errors

def validate_dataset_process(data: Dict[str, Any]) -> List[str]:
    """
    验证数据集处理请求
    
    Args:
        data: 数据集处理请求数据
        
    Returns:
        错误消息列表，如果没有错误则为空列表
    """
    errors = []
    
    # 检查必需字段
    if 'dataset_id' not in data:
        errors.append("缺少必需字段 'dataset_id'")
    elif not isinstance(data['dataset_id'], str):
        errors.append("'dataset_id' 必须是一个字符串")
    
    if 'output_name' not in data:
        errors.append("缺少必需字段 'output_name'")
    elif not isinstance(data['output_name'], str):
        errors.append("'output_name' 必须是一个字符串")
    elif not data['output_name'].strip():
        errors.append("'output_name' 不能为空")
    
    # 验证可选字段
    if 'train_ratio' in data:
        if not isinstance(data['train_ratio'], (int, float)):
            errors.append("'train_ratio' 必须是一个数字")
        elif data['train_ratio'] <= 0 or data['train_ratio'] > 1:
            errors.append("'train_ratio' 必须在 0 到 1 之间")
    
    if 'val_ratio' in data:
        if not isinstance(data['val_ratio'], (int, float)):
            errors.append("'val_ratio' 必须是一个数字")
        elif data['val_ratio'] < 0 or data['val_ratio'] >= 1:
            errors.append("'val_ratio' 必须在 0 到 1 之间")
    
    if 'test_ratio' in data:
        if not isinstance(data['test_ratio'], (int, float)):
            errors.append("'test_ratio' 必须是一个数字")
        elif data['test_ratio'] < 0 or data['test_ratio'] >= 1:
            errors.append("'test_ratio' 必须在 0 到 1 之间")
    
    # 检查比例总和
    if all(k in data for k in ['train_ratio', 'val_ratio', 'test_ratio']):
        total = data['train_ratio'] + data['val_ratio'] + data['test_ratio']
        if abs(total - 1.0) > 1e-6:
            errors.append("'train_ratio'、'val_ratio' 和 'test_ratio' 的总和必须为 1")
    
    if 'template' in data and not isinstance(data['template'], str):
        errors.append("'template' 必须是一个字符串")
    
    return errors
