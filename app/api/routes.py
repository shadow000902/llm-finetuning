from flask import request, jsonify, current_app
from functools import wraps
from datetime import datetime
import jwt

from app.api import bp
from config import Config

def token_required(f):
    """Authentication decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
            
        try:
            data = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=['HS256'])
        except:
            return jsonify({'error': 'Token is invalid'}), 401
            
        return f(*args, **kwargs)
    return decorated

@bp.route('/train', methods=['POST'])
@token_required
def train_model():
    """Start model training"""
    data = request.get_json()
    
    if not data or not data.get('dataset') or not data.get('config'):
        return jsonify({'error': 'Missing required parameters'}), 400
        
    try:
        training_id = current_app.training_service.start_training(
            data['config']
        )
        return jsonify({
            'message': 'Training started',
            'training_id': training_id
        }), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/generate', methods=['POST'])
@token_required
def generate_text():
    """Generate text from prompt"""
    data = request.get_json()
    
    if not data or not data.get('prompt'):
        return jsonify({'error': 'Prompt is required'}), 400
        
    try:
        result = current_app.model_service.generate_text(
            data['prompt'],
            max_length=data.get('max_length', 50),
            temperature=data.get('temperature', 0.7)
        )
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/models', methods=['GET'])
@token_required
def list_models():
    """List available model versions"""
    models = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    return jsonify([{
        'version': model.version,
        'path': model.path,
        'description': model.description,
        'created_at': model.created_at.isoformat(),
        'is_active': model.is_active
    } for model in models])

@bp.route('/models/<version>', methods=['PUT'])
@token_required
def activate_model(version):
    """Activate a specific model version"""
    model = ModelVersion.query.filter_by(version=version).first()
    if not model:
        return jsonify({'error': 'Model version not found'}), 404
        
    # Deactivate all other models
    ModelVersion.query.update({'is_active': False})
    
    # Activate selected model
    model.is_active = True
    db.session.commit()
    
    # Reload active model
    current_app.model_service.load_model(model.path)
    
    return jsonify({'message': f'Model {version} activated'})

@bp.route('/login', methods=['POST'])
def login():
    """User authentication"""
    data = request.get_json()
    # TODO: Implement actual authentication logic
    token = jwt.encode({
        'user_id': 1,
        'exp': datetime.utcnow() + Config.JWT_ACCESS_TOKEN_EXPIRES
    }, Config.JWT_SECRET_KEY, algorithm='HS256')
    
    return jsonify({'token': token})
