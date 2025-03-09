"""
模型数据访问对象模块
提供对模型相关数据库表的访问接口
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from app.extensions import db
from app.model import ModelVersion, TrainingRecord, EvaluationRecord

class ModelDAO:
    """模型数据访问对象，提供对模型相关表的CRUD操作"""

    @staticmethod
    def save_model_metadata(model_id: str, metadata: dict) -> bool:
        """保存模型元数据"""
        try:
            # 这里应该添加保存元数据的实际逻辑
            # 假设保存成功，返回True
            return True
        except Exception:
            # 若保存失败，返回False
            return False

    @staticmethod
    def get_model_metadata(model_id: str) -> dict:
        """获取模型元数据"""
        pass

    @staticmethod
    def update_model_status(model_id: str, status: str) -> bool:
        """更新模型训练状态"""
        try:
            # 这里应该添加更新状态的实际逻辑
            # 假设更新成功，返回True
            return True
        except Exception:
            # 若更新失败，返回False
            return False
    
    @staticmethod
    def create_model_version(version: str, path: str, description: Optional[str] = None, 
                           is_active: bool = False, training_record_id: Optional[int] = None) -> ModelVersion:
        """创建新的模型版本记录
        
        Args:
            version: 模型版本号
            path: 模型文件路径
            description: 模型描述
            is_active: 是否为激活版本
            training_record_id: 关联的训练记录ID
            
        Returns:
            创建的模型版本记录
        """
        model_version = ModelVersion(
            version=version,
            path=path,
            description=description,
            is_active=is_active,
            training_record_id=training_record_id
        )
        
        db.session.add(model_version)
        db.session.commit()
        return model_version
        
    @staticmethod
    def get_model_version(version_id: int) -> Optional[ModelVersion]:
        """根据ID获取模型版本
        
        Args:
            version_id: 模型版本ID
            
        Returns:
            模型版本记录，如果不存在则返回None
        """
        return ModelVersion.query.get(version_id)
        
    @staticmethod
    def get_model_version_by_version(version: str) -> Optional[ModelVersion]:
        """根据版本号获取模型版本
        
        Args:
            version: 模型版本号
            
        Returns:
            模型版本记录，如果不存在则返回None
        """
        return ModelVersion.query.filter_by(version=version).first()
        
    @staticmethod
    def get_active_model_version() -> Optional[ModelVersion]:
        """获取当前激活的模型版本
        
        Returns:
            激活的模型版本记录，如果不存在则返回None
        """
        return ModelVersion.query.filter_by(is_active=True).first()
        
    @staticmethod
    def activate_model_version(version_id: int) -> ModelVersion:
        """激活指定的模型版本
        
        Args:
            version_id: 要激活的模型版本ID
            
        Returns:
            激活的模型版本记录
            
        Raises:
            ValueError: 当模型版本不存在时抛出
        """
        # 获取要激活的模型版本
        model_version = ModelVersion.query.get(version_id)
        if not model_version:
            raise ValueError(f"Model version with ID {version_id} not found")
            
        # 取消所有模型版本的激活状态
        ModelVersion.query.update({'is_active': False})
        
        # 激活指定的模型版本
        model_version.is_active = True
        db.session.commit()
        
        return model_version
        
    @staticmethod
    def list_model_versions() -> List[ModelVersion]:
        """获取所有模型版本列表
        
        Returns:
            模型版本记录列表
        """
        return ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
        
    @staticmethod
    def create_training_record(config: Dict[str, Any]) -> TrainingRecord:
        """创建新的训练记录
        
        Args:
            config: 训练配置
            
        Returns:
            创建的训练记录
        """
        training_record = TrainingRecord(
            config=config,
            status='pending',
            start_time=datetime.utcnow()
        )
        
        db.session.add(training_record)
        db.session.commit()
        return training_record
        
    @staticmethod
    def update_training_record(record_id: int, status: str, progress: float = None, 
                             error_message: str = None) -> TrainingRecord:
        """更新训练记录状态
        
        Args:
            record_id: 训练记录ID
            status: 新状态
            progress: 训练进度
            error_message: 错误信息
            
        Returns:
            更新后的训练记录
            
        Raises:
            ValueError: 当训练记录不存在时抛出
        """
        training_record = TrainingRecord.query.get(record_id)
        if not training_record:
            raise ValueError(f"Training record with ID {record_id} not found")
            
        training_record.status = status
        
        if progress is not None:
            training_record.progress = progress
            
        if error_message is not None:
            training_record.error_message = error_message
            
        if status in ['completed', 'failed']:
            training_record.end_time = datetime.utcnow()
            
        db.session.commit()
        return training_record
        
    @staticmethod
    def create_evaluation_record(model_version_id: int, metrics: Dict[str, Any], 
                               dataset: str, notes: Optional[str] = None) -> EvaluationRecord:
        """创建新的评估记录
        
        Args:
            model_version_id: 模型版本ID
            metrics: 评估指标
            dataset: 数据集名称
            notes: 评估备注
            
        Returns:
            创建的评估记录
        """
        evaluation_record = EvaluationRecord(
            model_version_id=model_version_id,
            metrics=metrics,
            dataset=dataset,
            notes=notes
        )
        
        db.session.add(evaluation_record)
        db.session.commit()
        return evaluation_record
