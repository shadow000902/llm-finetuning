from datetime import datetime
from app import db

class TrainingRecord(db.Model):
    """Model for training records"""
    __tablename__ = 'training_records'
    
    id = db.Column(db.Integer, primary_key=True)
    config = db.Column(db.JSON, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    progress = db.Column(db.Float, default=0.0)
    error_message = db.Column(db.Text)
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'))
    
    def __repr__(self):
        return f'<TrainingRecord {self.id}>'

class ModelVersion(db.Model):
    """Model for versioned model instances"""
    __tablename__ = 'model_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50), unique=True, nullable=False)
    path = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=False)
    training_record_id = db.Column(db.Integer, db.ForeignKey('training_records.id'))
    
    training_record = db.relationship('TrainingRecord', backref='model_versions')
    
    def __repr__(self):
        return f'<ModelVersion {self.version}>'

class User(db.Model):
    """Model for system users"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<User {self.username}>'

class EvaluationRecord(db.Model):
    """Model for model evaluation records"""
    __tablename__ = 'evaluation_records'
    
    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey('model_versions.id'))
    metrics = db.Column(db.JSON, nullable=False)
    evaluated_at = db.Column(db.DateTime, default=datetime.utcnow)
    dataset = db.Column(db.String(255), nullable=False)
    notes = db.Column(db.Text)
    
    model_version = db.relationship('ModelVersion', backref='evaluations')
    
    def __repr__(self):
        return f'<EvaluationRecord {self.id}>'
