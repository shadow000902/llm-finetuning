from flask import Blueprint, current_app
from app.core.services.model_service import ModelService

bp = Blueprint('api', __name__)

# 使用 before_app_request 替代 before_app_first_request
# 添加一个标志变量来确保只初始化一次
_model_initialized = False

@bp.before_app_request
def initialize_model():
    global _model_initialized
    if not _model_initialized:
        # 使用应用中已初始化的 model_service
        # 从配置中获取模型路径
        model_path = current_app.config.get('MODEL_BASE_PATH', 'models') + '/model.pt'
        current_app.model_service.load_model(path=model_path)
        _model_initialized = True

from . import routes  # Import routes at the end to avoid circular imports
