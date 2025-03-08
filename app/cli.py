import os
import click
from flask.cli import FlaskGroup
from app import create_app
from config.config import config, Config
import argparse
import logging
import sys
from datetime import datetime
from app.model.training_service import ModelTrainingService
from app.utils.logging import setup_logging, get_logger

# 设置日志配置
setup_logging()

# 获取当前模块的日志记录器
logger = get_logger(__name__)

def get_app():
    """获取应用实例，支持环境变量配置"""
    env = os.getenv('FLASK_ENV', 'development')
    return create_app(config.get(env, config['default']))

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='DeepSeek-R1 1.5B 模型微调工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', '-c', required=True, help='训练配置文件路径')
    train_parser.add_argument('--log-file', '-l', help='日志文件路径')
    
    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='使用模型进行推理')
    infer_parser.add_argument('--model', '-m', required=True, help='模型路径')
    infer_parser.add_argument('--prompt', '-p', required=True, help='输入提示')
    infer_parser.add_argument('--max-length', type=int, default=100, help='最大生成长度')
    infer_parser.add_argument('--temperature', type=float, default=0.7, help='采样温度')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型性能')
    eval_parser.add_argument('--model', '-m', required=True, help='模型路径')
    eval_parser.add_argument('--dataset', '-d', required=True, help='评估数据集路径')
    eval_parser.add_argument('--output', '-o', help='评估结果输出路径')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'infer':
        # TODO: 实现推理命令
        logger.error("推理命令尚未实现")
        return 1
    elif args.command == 'evaluate':
        # TODO: 实现评估命令
        logger.error("评估命令尚未实现")
        return 1
    else:
        parser.print_help()
        return 0

@click.command('init-db')
@click.option('--force', is_flag=True, help='强制重新创建所有表')
def init_db_command(force):
    """初始化数据库命令"""
    from app.extensions import db
    
    if force:
        click.confirm('这将删除所有现有数据，确定要继续吗?', abort=True)
        db.drop_all()
        click.echo('已删除所有表')
    
    db.create_all()
    click.echo('数据库初始化完成')

@click.command('train-model')
@click.option('--config', required=True, help='训练配置文件路径')
@click.option('--output', help='模型输出路径')
def train_model_command(config, output):
    """训练模型命令"""
    from app.core.factories.service_factory import ServiceFactory
    import json
    
    with open(config, 'r') as f:
        config_data = json.load(f)
    
    if output:
        config_data['output_path'] = output
    
    training_service = ServiceFactory.create_training_service()
    result = training_service.start_training(config_data)
    click.echo(f'训练启动成功，训练ID: {result["training_id"]}')

@click.command('list-models')
def list_models_command():
    """列出所有可用模型"""
    from app.model import ModelVersion
    from app import create_app
    
    app = get_app()
    with app.app_context():
        models = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
        
        if not models:
            click.echo('没有找到任何模型')
            return
            
        click.echo('可用模型:')
        for model in models:
            status = '激活' if model.is_active else '未激活'
            click.echo(f'{model.version} - {status} - {model.description} - {model.created_at}')

def train_command(args):
    """处理训练命令
    
    Args:
        args: 命令行参数
    """
    try:
        # 创建训练服务
        training_service = ModelTrainingService()
        
        # 获取配置文件路径
        config_path = args.config
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            return 1
        
        # 创建任务ID
        task_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 设置日志文件
        if args.log_file:
            file_handler = logging.FileHandler(args.log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(file_handler)
            logger.info(f"日志将保存到: {args.log_file}")
        
        # 启动训练
        logger.info(f"开始训练任务 {task_id}，使用配置文件: {config_path}")
        result = training_service.train_model(config_path, task_id)
        
        # 输出结果
        if result['status'] == 'success':
            logger.info(f"训练成功完成！模型保存在: {result['model_path']}")
            logger.info(f"训练时间: {result['training_time']:.2f} 秒")
            return 0
        else:
            logger.error(f"训练失败: {result['error']}")
            return 1
            
    except Exception as e:
        logger.exception(f"训练过程中发生错误: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 