"""
项目启动脚本
负责初始化Flask应用并启动开发服务器
"""

import os
import argparse
from app import create_app
from app.utils.config_loader import load_config, get_config
from dotenv import load_dotenv

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLM Finetuning 应用服务器")
    parser.add_argument("--env", type=str, default=None, help="运行环境，默认从FLASK_ENV环境变量获取")
    parser.add_argument("--host", type=str, default=None, help="监听地址，默认从配置获取")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认从配置获取")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 确保环境变量已加载
    load_dotenv()
    
    # 确定环境
    env = args.env or os.getenv('FLASK_ENV', 'development')
    
    # 加载配置
    config = load_config(env)
    
    # 创建Flask应用实例
    app = create_app(env)
    
    # 确定运行参数
    host = args.host or config.get('app', {}).get('host', '0.0.0.0')
    port = args.port or config.get('app', {}).get('port', 5000)
    debug = args.debug or config.get('app', {}).get('debug', False)
    
    # 启动Flask开发服务器
    app.run(
        host=host,
        port=port,
        debug=debug
    )

if __name__ == '__main__':
    """
    主程序入口
    """
    main()
