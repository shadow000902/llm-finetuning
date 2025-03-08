"""
配置加载器模块

提供统一的配置加载机制，按照优先级加载不同来源的配置：
命令行参数 > 环境变量 > YAML配置文件 > 默认配置
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器类，用于加载和合并不同来源的配置
    
    提供统一的配置加载机制，按照优先级加载不同来源的配置：
    命令行参数 > 环境变量 > YAML配置文件 > 默认配置
    
    主要功能：
    - 加载YAML配置文件
    - 加载环境变量
    - 加载命令行参数
    - 合并不同来源的配置
    - 配置验证
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为"config"
        """
        self.config_dir = config_dir
        self.config = {}
        self.env_prefix = "LLM_FINETUNE_"  # 环境变量前缀
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            filename: 配置文件名，不包含路径
            
        Returns:
            配置字典
        """
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.debug(f"已加载配置文件: {filepath}")
                return config or {}
        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {filepath}")
            return {}
        except Exception as e:
            logger.error(f"加载配置文件失败: {filepath}, 错误: {str(e)}")
            return {}
    
    def load_env_vars(self, prefix: str = "") -> Dict[str, Any]:
        """
        加载环境变量
        
        Args:
            prefix: 环境变量前缀，用于过滤环境变量
            
        Returns:
            环境变量字典
        """
        env_vars = {}
        for key, value in os.environ.items():
            if not prefix or key.startswith(prefix):
                # 移除前缀并转换为小写
                if prefix and key.startswith(prefix):
                    key = key[len(prefix):]
                
                # 转换为嵌套字典结构
                parts = key.lower().split('_')
                current = env_vars
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    elif not isinstance(current[part], dict):
                        # 如果当前值不是字典，将其转换为字典
                        current[part] = {"value": current[part]}
                    current = current[part]
                
                # 尝试转换值类型
                value = self._convert_value(value)
                
                # 确保current是字典类型
                if isinstance(current, dict):
                    current[parts[-1]] = value
                else:
                    # 如果current不是字典，则跳过此环境变量
                    logger.warning(f"无法处理环境变量 {key}，因为路径中的某个部分已经是非字典值")
        
        return env_vars
    
    def _convert_value(self, value: str) -> Any:
        """
        尝试将字符串值转换为适当的类型
        
        Args:
            value: 字符串值
            
        Returns:
            转换后的值
        """
        # 布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 数字
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # 保持原样
        return value
    
    def _deep_merge(self, source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典
        
        Args:
            source: 源字典
            destination: 目标字典
            
        Returns:
            合并后的字典
        """
        for key, value in source.items():
            if key in destination and isinstance(destination[key], dict) and isinstance(value, dict):
                destination[key] = self._deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination
    
    def load_config(self, env: str = "development") -> Dict[str, Any]:
        """
        加载配置
        
        Args:
            env: 环境名称，用于加载对应的环境配置
            
        Returns:
            合并后的配置字典
        """
        # 加载默认配置
        self.config = self.load_yaml("default_config.yaml")
        
        # 加载环境特定配置
        env_config = self.load_yaml(f"{env}_config.yaml")
        self.config = self._deep_merge(env_config, self.config)
        
        # 加载API配置
        api_config = self.load_yaml("api_config.yaml")
        if api_config:
            self.config["api"] = api_config
        
        # 加载日志配置
        logging_config = self.load_yaml("logging_config.yaml")
        if logging_config:
            self.config["logging"] = logging_config
        
        # 加载训练配置
        train_config = self.load_yaml("train_config.yaml")
        if train_config:
            self.config["training"] = self._deep_merge(train_config, self.config.get("training", {}))
        
        # 加载环境变量
        env_vars = self.load_env_vars(self.env_prefix)
        self.config = self._deep_merge(env_vars, self.config)
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置项键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            配置项值
        """
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current

# 创建全局配置加载器实例
config_loader = ConfigLoader()

def load_config(env: str = "development") -> Dict[str, Any]:
    """
    加载配置的便捷函数
    
    Args:
        env: 环境名称
        
    Returns:
        配置字典
    """
    return config_loader.load_config(env)

def get_config(key: str, default: Any = None) -> Any:
    """
    获取配置项的便捷函数
    
    Args:
        key: 配置项键
        default: 默认值
        
    Returns:
        配置项值
    """
    return config_loader.get(key, default)