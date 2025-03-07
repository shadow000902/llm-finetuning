from typing import Dict, List, Optional
from flask import Request
from config import Config

def validate_generation_request(data: Dict) -> Optional[List[str]]:
    """Validate text generation request parameters"""
    errors = []
    
    if not data:
        return ['Request body is required']
        
    if 'prompt' not in data:
        errors.append('prompt is required')
    elif not isinstance(data['prompt'], str):
        errors.append('prompt must be a string')
        
    if 'max_length' in data and not isinstance(data['max_length'], int):
        errors.append('max_length must be an integer')
        
    if 'temperature' in data:
        if not isinstance(data['temperature'], (int, float)):
            errors.append('temperature must be a number')
        elif data['temperature'] < 0 or data['temperature'] > 2:
            errors.append('temperature must be between 0 and 2')
            
    return errors if errors else None

def validate_training_config(config: Dict) -> Optional[List[str]]:
    """Validate training configuration"""
    errors = []
    
    required_fields = [
        'dataset',
        'epochs',
        'batch_size'
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f'{field} is required')
            
    return errors if errors else None
