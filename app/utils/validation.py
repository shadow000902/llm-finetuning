from typing import Dict, List, Optional  # 导入类型注解相关类型
from flask import Request  # 导入Flask请求对象
from config.config import Config  # 导入配置类

def validate_generation_request(data: Dict) -> Optional[List[str]]:
    """验证文本生成请求参数
    
    Args:
        data (Dict): 请求参数字典
        
    Returns:
        Optional[List[str]]: 返回错误信息列表，如果没有错误则返回None
    """
    errors = []  # 初始化错误信息列表
    
    if not data:
        return ['Request body is required']  # 请求体不能为空
        
    if 'prompt' not in data:
        errors.append('prompt is required')  # prompt字段是必须的
    elif not isinstance(data['prompt'], str):
        errors.append('prompt must be a string')  # prompt必须是字符串类型
        
    if 'max_length' in data and not isinstance(data['max_length'], int):
        errors.append('max_length must be an integer')  # max_length必须是整数
        
    if 'temperature' in data:
        if not isinstance(data['temperature'], (int, float)):
            errors.append('temperature must be a number')  # temperature必须是数字
        elif data['temperature'] < 0 or data['temperature'] > 2:
            errors.append('temperature must be between 0 and 2')  # temperature必须在0到2之间
            
    return errors if errors else None  # 如果有错误返回错误列表，否则返回None

def validate_training_config(config: Dict) -> Optional[List[str]]:
    """验证训练配置
    
    Args:
        config (Dict): 训练配置字典
        
    Returns:
        Optional[List[str]]: 返回错误信息列表，如果没有错误则返回None
    """
    errors = []  # 初始化错误信息列表
    
    required_fields = [
        'dataset',  # 数据集路径
        'epochs',  # 训练轮数
        'batch_size'  # 批量大小
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f'{field} is required')  # 检查每个必填字段是否存在
            
    return errors if errors else None  # 如果有错误返回错误列表，否则返回None
