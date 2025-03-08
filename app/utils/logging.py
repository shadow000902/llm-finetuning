"""
日志工具模块
提供统一的日志配置和获取日志记录器的方法
"""

import logging
import os
import sys
import yaml
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler

def setup_logging(log_file: Optional[str] = None, log_level: int = None, log_format: str = None, config_file: str = None):
    """
    设置全局日志配置
    
    Args:
        log_file: 日志文件路径，如果为None则使用配置中的值
        log_level: 日志级别，如果为None则使用配置中的值
        log_format: 日志格式，如果为None则使用配置中的值
        config_file: 日志配置文件路径，如果提供则从文件加载配置
    """
    # 默认配置
    default_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": {
            "console": {
                "enabled": True,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "file": {
                "enabled": False,
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "filename": "logs/app.log",
                "max_bytes": 10485760,  # 10MB
                "backup_count": 5
            }
        },
        "third_party": {
            "transformers": "WARNING",
            "datasets": "WARNING",
            "torch": "WARNING",
            "accelerate": "WARNING"
        }
    }
    
    # 从配置文件加载配置
    config = default_config
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config = _merge_configs(config, file_config)
        except Exception as e:
            print(f"加载日志配置文件失败: {str(e)}")
    
    # 覆盖配置
    if log_level is not None:
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
        config["level"] = log_level
    else:
        config["level"] = getattr(logging, config["level"].upper())
        
    if log_format is not None:
        config["format"] = log_format
        
    if log_file is not None:
        config["handlers"]["file"]["enabled"] = True
        config["handlers"]["file"]["filename"] = log_file
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(config["level"])
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台日志处理器
    if config["handlers"]["console"]["enabled"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = getattr(logging, config["handlers"]["console"]["level"].upper())
        console_handler.setLevel(console_level)
        console_format = config["handlers"]["console"]["format"]
        console_handler.setFormatter(logging.Formatter(console_format))
        root_logger.addHandler(console_handler)
    
    # 添加文件日志处理器
    if config["handlers"]["file"]["enabled"]:
        file_config = config["handlers"]["file"]
        log_dir = os.path.dirname(file_config["filename"])
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            file_config["filename"],
            maxBytes=file_config["max_bytes"],
            backupCount=file_config["backup_count"]
        )
        file_level = getattr(logging, file_config["level"].upper())
        file_handler.setLevel(file_level)
        file_format = file_config["format"]
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    for lib, level in config["third_party"].items():
        lib_level = getattr(logging, level.upper())
        logging.getLogger(lib).setLevel(lib_level)
    
    return root_logger

def _merge_configs(default_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置
    
    Args:
        default_config: 默认配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = default_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result

def get_logger(name):
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称，通常使用__name__
        
    Returns:
        Logger: 日志记录器实例
    """
    return logging.getLogger(name) 