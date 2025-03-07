"""
interfaces包初始化文件
导出所有接口类，方便统一导入
"""

from .model_operations import IModelOperations
from .data_processing import IDataProcessor
from .training import ITrainingService

__all__ = [
    'IModelOperations',
    'IDataProcessor', 
    'ITrainingService'
]
