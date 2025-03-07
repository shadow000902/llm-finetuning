from datetime import datetime
from app import db

class TrainingRecord(db.Model):
    """Model for training records"""
    __tablename__ = 'training_records'
    
    id = db.Column(db.Integer, primary_key=True)  # 训练记录的唯一标识符
    config = db.Column(db.JSON, nullable=False)  # 训练配置参数，存储为JSON格式
    status = db.Column(db.String(20), nullable=False)  # 训练状态（如：running, completed, failed）
    start_time = db.Column(db.DateTime, default=datetime.utcnow)  # 训练开始时间，默认为当前UTC时间
    end_time = db.Column(db.DateTime)  # 训练结束时间
    progress = db.Column(db.Float, default=0.0)  # 训练进度，范围0.0到1.0
    error_message = db.Column(db.Text)  # 训练失败时的错误信息
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'))  # 关联的模型版本ID
    
    def __repr__(self):
        """返回训练记录的字符串表示，用于调试和日志记录"""
        return f'<TrainingRecord {self.id}>'

class ModelVersion(db.Model):
    """Model for versioned model instances"""
    __tablename__ = 'model_versions'
    
    id = db.Column(db.Integer, primary_key=True)  # 模型版本的唯一标识符
    version = db.Column(db.String(50), unique=True, nullable=False)  # 模型版本号，必须唯一
    path = db.Column(db.String(255), nullable=False)  # 模型文件的存储路径
    description = db.Column(db.Text)  # 模型版本的描述信息
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 创建时间，默认为当前UTC时间
    is_active = db.Column(db.Boolean, default=False)  # 标记是否为当前激活版本
    training_record_id = db.Column(db.Integer, db.ForeignKey('training_records.id'))  # 关联的训练记录ID
    
    training_record = db.relationship('TrainingRecord', backref='model_versions')  # 与TrainingRecord的一对多关系
    
    def __repr__(self):
        """返回模型版本的字符串表示，用于调试和日志记录"""
        return f'<ModelVersion {self.version}>'

class User(db.Model):
    """Model for system users"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)  # 用户的唯一标识符
    username = db.Column(db.String(50), unique=True, nullable=False)  # 用户名，必须唯一
    password_hash = db.Column(db.String(128), nullable=False)  # 密码的哈希值
    is_admin = db.Column(db.Boolean, default=False)  # 标记是否为管理员用户
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 用户创建时间，默认为当前UTC时间
    last_login = db.Column(db.DateTime)  # 用户最后登录时间
    
    def __repr__(self):
        """返回用户的字符串表示，用于调试和日志记录"""
        return f'<User {self.username}>'

class EvaluationRecord(db.Model):
    """Model for model evaluation records"""
    __tablename__ = 'evaluation_records'
    
    id = db.Column(db.Integer, primary_key=True)  # 评估记录的唯一标识符
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'))  # 关联的模型版本ID
    metrics = db.Column(db.JSON, nullable=False)  # 评估指标，存储为JSON格式
    evaluated_at = db.Column(db.DateTime, default=datetime.utcnow)  # 评估时间，默认为当前UTC时间
    dataset = db.Column(db.String(255), nullable=False)  # 用于评估的数据集名称或路径
    notes = db.Column(db.Text)  # 评估备注信息
    
    model_version = db.relationship('ModelVersion', backref='evaluations')  # 与ModelVersion的一对多关系
    
    def __repr__(self):
        """返回评估记录的字符串表示，用于调试和日志记录"""
        return f'<EvaluationRecord {self.id}>'
