from flask import jsonify
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
import re

def setup_swagger(app):
    """
    设置Swagger文档
    
    Args:
        app: Flask应用实例
    """
    # 获取API前缀
    api_prefix = app.config['APP_CONFIG'].get('api', {}).get('prefix', '/api/v1')
    
    # 设置Swagger UI蓝图
    SWAGGER_URL = '/swagger'  # Swagger UI的URL
    API_URL = '/swagger.json'  # Swagger JSON文件的URL
    
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            'app_name': "LLM微调项目API文档"
        }
    )
    
    # 注册Swagger UI蓝图
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
    
    # 添加swagger.json路由
    @app.route('/swagger.json')
    def get_swagger():
        swag = swagger(app)
        swag['info'] = {
            'title': 'LLM微调项目API',
            'description': 'LLM微调项目的RESTful API文档',
            'version': app.config.get('VERSION', '0.1.0'),
            'termsOfService': '',
            'contact': {
                'name': 'API Support',
                'email': 'support@example.com'
            },
            'license': {
                'name': 'MIT',
                'url': 'https://opensource.org/licenses/MIT'
            }
        }
        
        # 添加安全定义
        swag['securityDefinitions'] = {
            'Bearer': {
                'type': 'apiKey',
                'name': 'Authorization',
                'in': 'header',
                'description': '使用Bearer令牌进行身份验证，格式: Bearer {token}'
            }
        }
        
        # 处理路由中的请求体定义
        process_request_bodies(app, swag)

        return jsonify(swag)

def process_request_bodies(app, swag):
    """
    处理API路由中的请求体定义，将其添加到Swagger文档中
    
    Args:
        app: Flask应用实例
        swag: Swagger文档对象
    """
    # 确保paths存在
    if 'paths' not in swag:
        return
    
    # 遍历所有路径
    for path, path_info in swag['paths'].items():
        # 遍历所有HTTP方法
        for method, method_info in path_info.items():
            # 只处理POST、PUT等可能有请求体的方法
            if method.lower() in ['post', 'put', 'patch']:
                # 获取视图函数
                view_func = app.view_functions.get(method_info.get('operationId'))
                if not view_func:
                    continue
                
                # 获取docstring
                docstring = view_func.__doc__
                if not docstring:
                    continue
                
                # 解析docstring中的requestBody部分
                request_body = extract_request_body(docstring)
                if request_body:
                    method_info['parameters'] = method_info.get('parameters', [])
                    
                    # 添加请求体参数
                    # 使用consumes和produces字段指定内容类型
                    method_info['consumes'] = ['application/json']
                    method_info['produces'] = ['application/json']
                    
                    # 直接设置requestBody字段，而不是使用parameters
                    method_info['requestBody'] = {
                        'description': request_body.get('description', '请求体参数'),
                        'required': request_body.get('required', True),
                        'content': {
                            'application/json': {
                                'schema': request_body.get('schema', {})
                            }
                        }
                    }
                    
                    # 不再将body参数添加到parameters数组中
                    # OpenAPI 3.0使用requestBody而不是parameters来定义请求体
                    # 保留parameters数组用于路径参数、查询参数等

def extract_request_body(docstring):
    """
    从docstring中提取requestBody部分
    
    Args:
        docstring: 函数文档字符串
        
    Returns:
        dict: 请求体定义，如果没有则返回None
    """
    if not docstring:
        return None
    
    # 查找requestBody部分
    match = re.search(r'requestBody:\s*\n(.*?)(?:responses:|\Z)', docstring, re.DOTALL)
    if not match:
        return None
    
    request_body_text = match.group(1)
    
    # 解析required
    required = 'required: true' in request_body_text
    
    # 解析schema
    schema = {}
    
    # 查找content部分中的schema
    content_schema_match = re.search(r'content:\s*\n.*?application/json:\s*\n.*?schema:\s*\n(.*?)(?:responses:|\Z)', request_body_text, re.DOTALL)
    if content_schema_match:
        schema_text = content_schema_match.group(1)
        
        # 解析type
        type_match = re.search(r'type:\s*(\w+)', schema_text)
        if type_match:
            schema['type'] = type_match.group(1)
        
        # 解析properties
        if 'properties:' in schema_text:
            schema['properties'] = {}
            
            # 提取属性定义 - 改进正则表达式以更好地匹配属性定义
            properties_match = re.search(r'properties:\s*\n(.*?)(?:(?:^\s*required:)|responses:|\Z)', schema_text, re.DOTALL | re.MULTILINE)
            if properties_match:
                properties_text = properties_match.group(1)
                
                # 改进属性匹配的正则表达式，使其能够更准确地匹配缩进的属性定义
                property_matches = re.finditer(r'\s+(\w+):\s*\n((?:\s+[^\n]+\n)+)', properties_text)
                for property_match in property_matches:
                    property_name = property_match.group(1)
                    property_text = property_match.group(2)
                    
                    property_def = {}
                    
                    # 解析属性类型
                    type_match = re.search(r'type:\s*(\w+)', property_text)
                    if type_match:
                        property_def['type'] = type_match.group(1)
                    
                    # 解析描述
                    desc_match = re.search(r'description:\s*(.+)', property_text)
                    if desc_match:
                        property_def['description'] = desc_match.group(1).strip()
                    
                    # 解析格式
                    format_match = re.search(r'format:\s*(.+)', property_text)
                    if format_match:
                        property_def['format'] = format_match.group(1).strip()
                    
                    # 解析默认值
                    default_match = re.search(r'default:\s*(.+)', property_text)
                    if default_match:
                        default_value = default_match.group(1).strip()
                        try:
                            # 尝试转换为数字
                            if default_value.replace('.', '', 1).isdigit():
                                if '.' in default_value:
                                    property_def['default'] = float(default_value)
                                else:
                                    property_def['default'] = int(default_value)
                            elif default_value.lower() in ['true', 'false']:
                                property_def['default'] = default_value.lower() == 'true'
                            else:
                                property_def['default'] = default_value
                        except:
                            property_def['default'] = default_value
                    
                    schema['properties'][property_name] = property_def
        
        # 解析required属性列表 - 改进正则表达式以更好地匹配required属性
        required_props_match = re.search(r'required:\s*\n(.*?)(?=\n\s*\w+:|\Z)', schema_text, re.DOTALL)
        if required_props_match:
            required_text = required_props_match.group(1)
            schema['required'] = [item.strip() for item in re.findall(r'-\s*(\w+)', required_text)]
        else:
            # 尝试匹配直接列出的required属性（不使用破折号格式）
            direct_required_match = re.search(r'required:\s*\n\s*-\s*([\w\s-]+)', schema_text, re.DOTALL) or re.search(r'required:\s*\n((?:\s*-\s*\w+\s*\n)+)', schema_text, re.DOTALL)
            if direct_required_match:
                required_text = direct_required_match.group(1)
                schema['required'] = [item.strip() for item in re.findall(r'-\s*(\w+)', required_text)]
    
    return {
        'required': required,
        'schema': schema,
        'description': '请求体参数'
    }