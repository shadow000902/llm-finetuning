import uuid
from datetime import datetime, timedelta
from functools import wraps

import jwt
from flask import request, jsonify, current_app, g, render_template
from werkzeug.utils import secure_filename

from app.api import bp
from app.model import ModelVersion
from app.utils.validation import (
    validate_generation_request,
    validate_training_config,
    validate_batch_generation_request,
    validate_dataset_upload,
    validate_dataset_process
)


# 获取API配置
def get_api_config():
    """获取API配置"""
    return current_app.config['APP_CONFIG'].get('api', {})

# API文档路由
@bp.route('/docs', methods=['GET'])
def api_docs():
    """API文档页面"""
    # 检查是否启用文档
    if not get_api_config().get('enable_docs', True):
        return jsonify({
            'error': {
                'code': 'NOT_FOUND',
                'message': 'API documentation is disabled'
            }
        }), 404
        
    return render_template('api_docs.html')

def token_required(f):
    """认证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # 检查是否需要令牌
        if not get_api_config().get('token_required', True):
            return f(*args, **kwargs)
            
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
            else:
                token = auth_header
                
        if not token:
            return jsonify({'error': 'Token is missing', 'code': 'UNAUTHORIZED'}), 401
            
        try:
            # 使用配置中的JWT密钥
            jwt_secret = current_app.config['APP_CONFIG'].get('security', {}).get('jwt_secret_key', 'jwt-secret-key')
            data = jwt.decode(token, jwt_secret, algorithms=['HS256'])
            g.user_id = data.get('user_id')
        except:
            return jsonify({'error': 'Token is invalid', 'code': 'UNAUTHORIZED'}), 401
            
        return f(*args, **kwargs)
    return decorated

def rate_limit(limit_key):
    """速率限制装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # 检查是否启用速率限制
            if not get_api_config().get('enable_rate_limit', False):
                return f(*args, **kwargs)
                
            # 如果使用了Flask-Limiter，则应用速率限制
            if 'limiter' in current_app.extensions:
                limiter = current_app.extensions['limiter']
                limit = get_api_config().get(f'rate_limit_{limit_key}', get_api_config().get('rate_limit_default', '100/minute'))
                
                # 动态设置速率限制
                limiter.limit(limit)(f)
                
            return f(*args, **kwargs)
        return decorated
    return decorator

# 1. 模型训练API

@bp.route('/train', methods=['POST'])
@token_required
@rate_limit('train')
def train_model():
    """启动模型训练"""
    data = request.get_json()
    
    # 验证请求数据
    errors = validate_training_config(data)
    if errors:
        return jsonify({
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Invalid training configuration',
                'details': errors
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 400
        
    try:
        training_id = current_app.training_service.start_training(
            data
        )
        return jsonify({
            'task_id': training_id,
            'status': 'started',
            'message': '训练任务已启动',
            'created_at': datetime.utcnow().isoformat()
        }), 202
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

@bp.route('/train/status/<task_id>', methods=['GET'])
@token_required
def training_status(task_id):
    """获取训练状态"""
    try:
        status = current_app.training_service.get_training_status()
        if task_id in status:
            return jsonify(status[task_id])
        return jsonify({
            'error': {
                'code': 'NOT_FOUND',
                'message': 'Training task not found'
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 404
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

@bp.route('/train/cancel/<task_id>', methods=['POST'])
@token_required
def cancel_training(task_id):
    """取消训练任务"""
    try:
        success = current_app.training_service.cancel_training(task_id)
        if success:
            return jsonify({
                'task_id': task_id,
                'status': 'canceled',
                'message': '训练任务已取消',
                'canceled_at': datetime.utcnow().isoformat()
            })
        return jsonify({
            'error': {
                'code': 'NOT_FOUND',
                'message': 'Training task not found'
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 404
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

# 2. 模型推理API

@bp.route('/generate', methods=['POST'])
@token_required
@rate_limit('inference')
def generate_text():
    """生成文本"""
    data = request.get_json()
    
    # 验证请求数据
    errors = validate_generation_request(data)
    if errors:
        return jsonify({
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Invalid generation request',
                'details': errors
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 400
        
    try:
        start_time = datetime.utcnow()
        
        # 获取模型路径
        model_path = data.get('model_path')
        
        # 生成文本
        result = current_app.model_service.generate_text(
            data['prompt'],
            model_path=model_path,
            max_length=data.get('max_length', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            top_k=data.get('top_k', 50),
            num_return_sequences=data.get('num_return_sequences', 1)
        )
        
        end_time = datetime.utcnow()
        generation_time = (end_time - start_time).total_seconds()
        
        return jsonify({
            'prompt': data['prompt'],
            'generated_texts': result,
            'model_path': model_path or current_app.model_service.get_active_model_path(),
            'generation_time': generation_time,
            'timestamp': end_time.isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

@bp.route('/generate/batch', methods=['POST'])
@token_required
@rate_limit('inference')
def generate_batch():
    """批量生成文本"""
    data = request.get_json()
    
    # 验证请求数据
    errors = validate_batch_generation_request(data)
    if errors:
        return jsonify({
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Invalid batch generation request',
                'details': errors
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 400
        
    try:
        start_time = datetime.utcnow()
        
        # 获取模型路径和生成参数
        model_path = data.get('model_path')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        top_k = data.get('top_k', 50)
        num_return_sequences = data.get('num_return_sequences', 1)
        
        # 批量生成文本
        results = []
        for prompt in data['prompts']:
            result = current_app.model_service.generate_text(
                prompt,
                model_path=model_path,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences
            )
            
            results.append({
                'prompt': prompt,
                'generated_texts': result
            })
        
        end_time = datetime.utcnow()
        total_generation_time = (end_time - start_time).total_seconds()
        
        return jsonify({
            'results': results,
            'model_path': model_path or current_app.model_service.get_active_model_path(),
            'total_generation_time': total_generation_time,
            'timestamp': end_time.isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

# 3. 模型管理API

@bp.route('/models', methods=['GET'])
@token_required
def list_models():
    """获取可用模型列表"""
    try:
        models = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
        return jsonify({
            'models': [{
                'id': model.id,
                'name': model.name,
                'path': model.path,
                'base_model': model.base_model,
                'created_at': model.created_at.isoformat(),
                'size_mb': model.size_mb,
                'description': model.description
            } for model in models],
            'count': len(models),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

@bp.route('/models/<model_id>', methods=['GET'])
@token_required
def get_model(model_id):
    """获取模型详情"""
    try:
        model = ModelVersion.query.filter_by(id=model_id).first()
        if not model:
            return jsonify({
                'error': {
                    'code': 'NOT_FOUND',
                    'message': 'Model not found'
                },
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': str(uuid.uuid4())
            }), 404
            
        # 获取模型详情
        metrics = current_app.model_service.get_model_metrics(model.path)
        training_config = current_app.model_service.get_model_config(model.path)
        usage_stats = current_app.model_service.get_model_usage_stats(model.id)
        
        return jsonify({
            'id': model.id,
            'name': model.name,
            'path': model.path,
            'base_model': model.base_model,
            'created_at': model.created_at.isoformat(),
            'size_mb': model.size_mb,
            'description': model.description,
            'training_config': training_config,
            'metrics': metrics,
            'last_used': usage_stats.get('last_used'),
            'usage_count': usage_stats.get('usage_count', 0)
        })
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

# 4. 数据集管理API

@bp.route('/datasets', methods=['GET'])
@token_required
def list_datasets():
    """获取数据集列表"""
    try:
        datasets = current_app.data_service.list_datasets()
        return jsonify({
            'datasets': datasets,
            'count': len(datasets),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

@bp.route('/datasets/upload', methods=['POST'])
@token_required
def upload_dataset():
    """上传数据集"""
    # 验证请求数据
    errors = validate_dataset_upload(request)
    if errors:
        return jsonify({
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Invalid dataset upload request',
                'details': errors
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 400
        
    try:
        # 获取文件和元数据
        file = request.files['file']
        name = request.form.get('name')
        description = request.form.get('description', '')
        format = request.form.get('format', 'json')
        
        # 安全地获取文件名
        filename = secure_filename(file.filename)
        
        # 保存文件
        dataset_id = current_app.data_service.save_dataset(
            file=file,
            name=name,
            description=description,
            format=format,
            filename=filename
        )
        
        # 获取数据集信息
        dataset_info = current_app.data_service.get_dataset_info(dataset_id)
        
        return jsonify({
            'id': dataset_id,
            'name': name,
            'path': dataset_info.get('path'),
            'uploaded_at': datetime.utcnow().isoformat(),
            'size_mb': dataset_info.get('size_mb'),
            'samples_count': dataset_info.get('samples_count'),
            'description': description,
            'message': '数据集上传成功'
        })
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

@bp.route('/datasets/process', methods=['POST'])
@token_required
def process_dataset():
    """处理数据集"""
    data = request.get_json()
    
    # 验证请求数据
    errors = validate_dataset_process(data)
    if errors:
        return jsonify({
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Invalid dataset process request',
                'details': errors
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 400
        
    try:
        # 处理数据集
        task_id = current_app.data_service.process_dataset(
            dataset_id=data['dataset_id'],
            output_name=data['output_name'],
            train_ratio=data.get('train_ratio', 0.8),
            val_ratio=data.get('val_ratio', 0.1),
            test_ratio=data.get('test_ratio', 0.1),
            template=data.get('template')
        )
        
        # 获取处理结果
        result = current_app.data_service.get_processing_result(task_id)
        
        return jsonify({
            'task_id': task_id,
            'status': result.get('status', 'completed'),
            'dataset_id': data['dataset_id'],
            'output_path': result.get('output_path'),
            'splits': result.get('splits', {}),
            'message': '数据集处理完成',
            'processed_at': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 500

# 5. 认证API

@bp.route('/login', methods=['POST'])
def login():
    """用户认证"""
    data = request.get_json()
    
    # 简单的用户验证逻辑，实际应用中应该查询数据库
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Username and password are required'
            },
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': str(uuid.uuid4())
        }), 400
        
    # TODO: 实现实际的用户验证逻辑
    # 这里仅作为示例，应该替换为实际的用户验证
    if data.get('username') == 'admin' and data.get('password') == 'password':
        # 获取JWT配置
        jwt_secret = current_app.config['APP_CONFIG'].get('security', {}).get('jwt_secret_key', 'jwt-secret-key')
        jwt_expires = current_app.config['APP_CONFIG'].get('security', {}).get('jwt_access_token_expires', 3600)
        
        # 生成令牌
        token = jwt.encode({
            'user_id': 1,
            'exp': datetime.utcnow() + timedelta(seconds=jwt_expires)
        }, jwt_secret, algorithm='HS256')
        
        return jsonify({
            'token': token,
            'expires_in': jwt_expires,
            'token_type': 'Bearer'
        })
    
    return jsonify({
        'error': {
            'code': 'UNAUTHORIZED',
            'message': 'Invalid credentials'
        },
        'timestamp': datetime.utcnow().isoformat(),
        'request_id': str(uuid.uuid4())
    }), 401
