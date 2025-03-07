from app import db  # 导入数据库实例
from app.model import ModelVersion, TrainingRecord  # 导入模型版本和训练记录模型
from datetime import datetime  # 导入日期时间处理模块

class ModelDAO:
    """模型数据访问对象，负责处理与模型相关的数据库操作"""
    @staticmethod
    def create_training_record(config):
        """创建新的训练记录
        
        Args:
            config (dict): 训练配置参数
            
        Returns:
            TrainingRecord: 新创建的训练记录对象
        """
        # 创建训练记录对象
        record = TrainingRecord(
            config=config,  # 训练配置
            status='initializing',  # 初始状态为初始化中
            start_time=datetime.utcnow()  # 记录当前UTC时间作为开始时间
        )
        db.session.add(record)  # 将记录添加到数据库会话
        db.session.commit()  # 提交事务
        return record  # 返回创建的记录对象

    @staticmethod
    def update_training_record(record_id, status, error_message=None):
        """更新训练记录状态
        
        Args:
            record_id (int): 要更新的训练记录ID
            status (str): 新的状态值
            error_message (str, optional): 错误信息，如果有的话
            
        Returns:
            TrainingRecord: 更新后的训练记录对象
            
        Raises:
            ValueError: 如果找不到对应的训练记录
        """
        # 根据ID查询训练记录
        record = TrainingRecord.query.get(record_id)
        if not record:
            raise ValueError(f'Training record {record_id} not found')
            
        # 更新记录状态
        record.status = status
        record.end_time = datetime.utcnow()  # 记录当前UTC时间作为结束时间
        if error_message:
            record.error_message = error_message  # 如果有错误信息则记录
        db.session.commit()  # 提交事务
        return record  # 返回更新后的记录对象

    @staticmethod
    def create_model_version(version, path, description=None):
        """创建新的模型版本记录
        
        Args:
            version (str): 模型版本号
            path (str): 模型文件存储路径
            description (str, optional): 版本描述信息
            
        Returns:
            ModelVersion: 新创建的模型版本对象
        """
        # 创建模型版本对象
        version = ModelVersion(
            version=version,  # 模型版本号
            path=path,  # 模型文件存储路径
            description=description  # 版本描述信息
        )
        db.session.add(version)  # 将版本记录添加到数据库会话
        db.session.commit()  # 提交事务
        return version  # 返回创建的版本对象
