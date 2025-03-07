"""
应用扩展模块
负责初始化和管理Flask应用的扩展
当前主要包含SQLAlchemy数据库扩展
"""

from flask_sqlalchemy import SQLAlchemy

# SQLAlchemy数据库扩展实例
# 用于管理数据库连接和ORM操作
db = SQLAlchemy()
