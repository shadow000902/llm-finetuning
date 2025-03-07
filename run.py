"""
项目启动脚本
负责初始化Flask应用并启动开发服务器
"""

import os
from app import create_app
from config import Config

# 创建Flask应用实例
app = create_app(Config)

if __name__ == '__main__':
    """
    主程序入口
    """
    # 确保环境变量已加载
    from dotenv import load_dotenv
    load_dotenv()
    
    # 启动Flask开发服务器
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),  # 监听地址，默认所有网络接口
        port=int(os.getenv('FLASK_PORT', '5000')),  # 监听端口，默认5000
        debug=Config.FLASK_DEBUG  # 调试模式，从配置中读取
    )
