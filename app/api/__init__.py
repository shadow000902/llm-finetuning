from flask import Blueprint
from app.core.services.model_service import ModelService

bp = Blueprint('api', __name__)
model_service = ModelService()

@bp.before_app_first_request
def initialize_model():
    model_service.load_model()

from . import routes  # Import routes at the end to avoid circular imports
