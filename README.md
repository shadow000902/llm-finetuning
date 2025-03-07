# LLM Finetuning Project

基于DeepSeek-R1 1.5B模型进行业务方向调优的工业级项目

## 项目背景
本项目旨在为企业提供定制化的大语言模型微调解决方案，基于DeepSeek-R1 1.5B模型，通过业务数据微调实现特定领域的性能优化。

## 功能特性
- 支持模型微调训练
- 提供RESTful API接口
- 完整的训练监控和评估
- 可配置的训练参数
- 支持分布式训练
- 模型版本管理

## 技术栈
- Python 3.9+
- PyTorch
- Flask
- HuggingFace Transformers
- Docker
- Redis
- Celery

## 安装指南

### 环境要求
- Python 3.9+
- CUDA 11.7+ (GPU训练)
- Docker (可选)

### 安装步骤
1. 克隆项目仓库
```bash
git clone https://github.com/your-repo/llm-finetuning.git
cd llm-finetuning
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
复制.env.example文件为.env，并根据实际情况修改配置

## 快速开始

### 启动开发服务器
```bash
python run.py
```

### 训练模型
```bash
python -m app.model.training_service --config config/train_config.yaml
```

### API文档
启动服务后访问 `/docs` 查看API文档

## 项目结构
```
.
├── app/                  # 应用核心代码
│   ├── api/              # API接口
│   ├── core/             # 核心业务逻辑
│   ├── model/            # 模型相关实现
│   └── utils/            # 工具类
├── config/               # 配置文件
├── tests/                # 测试代码
├── requirements.txt      # 依赖文件
└── run.py                # 启动脚本
```

## 贡献指南
欢迎贡献代码！请遵循以下步骤：
1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

## 许可证
本项目采用 MIT 许可证 - 详情请见 LICENSE 文件
