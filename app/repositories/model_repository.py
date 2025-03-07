from app import db
from llm_finetuning_project.domain.models import ModelVersion, TrainingRecord
from datetime import datetime

class ModelDAO:
    @staticmethod
    def create_training_record(config):
        """Create new training record"""
        record = TrainingRecord(
            config=config,
            status='initializing',
            start_time=datetime.utcnow()
        )
        db.session.add(record)
        db.session.commit()
        return record

    @staticmethod
    def update_training_record(record_id, status, error_message=None):
        """Update training record status"""
        record = TrainingRecord.query.get(record_id)
        if not record:
            raise ValueError(f'Training record {record_id} not found')
            
        record.status = status
        record.end_time = datetime.utcnow()
        if error_message:
            record.error_message = error_message
        db.session.commit()
        return record

    @staticmethod
    def create_model_version(version, path, description=None):
        """Create new model version record"""
        version = ModelVersion(
            version=version,
            path=path,
            description=description
        )
        db.session.add(version)
        db.session.commit()
        return version
