# LLM 微调项目

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
git clone https://github.com/shadow000902/llm-finetuning.git
cd llm-finetuning
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate  # Windows
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
python -m app.cli train --config config/train_config.yaml --log-file logs/training.log
```

### 使用API
启动服务后，可以通过以下API进行操作：

#### 启动训练任务
```bash
curl -X POST http://localhost:5000/api/v1/train \
  -H "Content-Type: application/json" \
  -H "Authorization: YOUR_TOKEN" \
  -d '{
    "dataset_path": "/data/train.json",
    "config": {
      "learning_rate": 3e-5,
      "batch_size": 16,
      "num_epochs": 3
    }
  }'
```

#### 获取训练状态
```bash
curl -X GET http://localhost:5000/api/v1/train/status/{task_id} \
  -H "Authorization: YOUR_TOKEN"
```

#### 模型推理
```bash
curl -X POST http://localhost:5000/api/v1/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: YOUR_TOKEN" \
  -d '{
    "prompt": "请解释一下量子计算的基本原理",
    "max_length": 100,
    "temperature": 0.7
  }'
```

## 项目结构
```
.
├── .env                  # 环境变量配置文件
├── .gitignore            # Git忽略文件配置
├── README.md             # 项目说明文档
├── requirements.txt      # Python依赖文件
├── run.py                # 项目启动脚本
├── app/                  # 应用核心代码
│   ├── __init__.py       # 应用初始化
│   ├── api/              # API接口
│   │   ├── __init__.py   # API模块初始化
│   │   └── routes.py     # API路由配置
│   ├── core/             # 核心业务逻辑
│   ├── model/            # 模型相关实现
│   ├── repositories/     # 数据仓库
│   └── utils/            # 工具类
├── config/               # 配置文件
│   ├── config.py         # 配置管理
│   ├── train_config.yaml # 训练配置示例
│   └── prod_config.yaml  # 生产环境配置
├── logs/                 # 日志文件目录
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后的数据
│   └── embeddings/       # 嵌入向量数据
├── models/               # 模型存储目录
│   ├── checkpoints/      # 模型检查点
│   └── exports/          # 导出模型
└── tests/                # 测试代码
```

## 常见问题解答

**Q: 如何选择合适的batch size？**
A: 建议从16开始，根据GPU显存逐步增加，直到达到显存上限的80%

**Q: 训练过程中出现OOM错误怎么办？**
A: 可以尝试以下方法：
   - 减小batch size
   - 启用梯度检查点（gradient checkpointing）
   - 使用混合精度训练

**Q: 如何评估模型效果？**
A: 可以使用以下指标：
   - 困惑度（Perplexity）
   - BLEU分数
   - 人工评估

## 贡献指南
欢迎贡献代码！请遵循以下步骤：
1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

## 许可证
本项目采用 MIT 许可证 - 详情请见 LICENSE 文件 