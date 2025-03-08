# 配置说明

本文档详细说明了项目的配置选项，包括环境变量、训练配置和API配置等。

## 环境变量配置

项目使用`.env`文件存储环境变量配置。以下是可用的环境变量及其说明：

### 基本配置

```
# 应用基本配置
APP_NAME=LLM Finetuning
APP_ENV=development  # development, testing, production
APP_DEBUG=true
APP_PORT=5000
APP_HOST=0.0.0.0
APP_SECRET_KEY=your-secret-key-here
```

### 日志配置

```
# 日志配置
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_TO_FILE=true
LOG_TO_CONSOLE=true
LOG_FILE=logs/app.log
```

### 数据库配置

```
# 数据库配置（可选）
DB_TYPE=sqlite  # sqlite, mysql, postgresql
DB_HOST=localhost
DB_PORT=3306
DB_NAME=llm_finetuning
DB_USER=root
DB_PASSWORD=password
```

### Redis配置

```
# Redis配置（用于任务队列）
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### Celery配置

```
# Celery配置（用于异步任务）
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
CELERY_ACCEPT_CONTENT=json
CELERY_TIMEZONE=Asia/Shanghai
```

### 模型配置

```
# 模型配置
DEFAULT_MODEL_PATH=deepseek-ai/deepseek-llm-1.5b-base
MODEL_CACHE_DIR=./models/cache
MODEL_CHECKPOINTS_DIR=./models/checkpoints
MODEL_EXPORTS_DIR=./models/exports
```

### API配置

```
# API配置
API_TOKEN=your-api-token-here
API_RATE_LIMIT_TRAIN=10/hour
API_RATE_LIMIT_INFERENCE=60/minute
API_RATE_LIMIT_DEFAULT=120/minute
```

## 训练配置

训练配置使用YAML文件格式，通常存储在`config/train_config.yaml`文件中。以下是配置选项及其说明：

### 模型配置

```yaml
model:
  # 基础模型路径或名称
  base_model: "deepseek-ai/deepseek-llm-1.5b-base"
  
  # 最大序列长度
  max_length: 512
  
  # 量化类型，可选值为"int8"、"int4"或null
  quantization: "int8"
```

### LoRA配置

```yaml
lora:
  # LoRA秩，控制可训练参数的数量
  r: 8
  
  # LoRA alpha参数，通常设置为r的2倍
  lora_alpha: 16
  
  # LoRA dropout率，用于防止过拟合
  lora_dropout: 0.05
  
  # 偏置项处理方式，可选值为"none"、"all"或"lora_only"
  bias: "none"
```

### 训练参数

```yaml
training:
  # 每个设备的训练批次大小
  batch_size: 8
  
  # 梯度累积步数，用于模拟更大的批次大小
  gradient_accumulation_steps: 1
  
  # 训练轮数
  num_epochs: 3
  
  # 学习率
  learning_rate: 3e-4
  
  # 是否使用混合精度训练（FP16）
  fp16: true
  
  # 日志记录步数
  logging_steps: 100
  
  # 模型保存步数
  save_steps: 1000
  
  # 评估步数
  eval_steps: 500
  
  # 保存的检查点数量限制
  save_total_limit: 3
  
  # 是否加载验证集上表现最好的模型
  load_best_model_at_end: true
  
  # 报告工具，可选值为"tensorboard"、"wandb"等
  report_to: "tensorboard"
  
  # 权重衰减率
  weight_decay: 0.01
  
  # 预热步数
  warmup_steps: 500
  
  # 学习率调度器类型
  lr_scheduler_type: "cosine"
```

### 输出配置

```yaml
# 输出目录
output_dir: "./models/checkpoints/sample"
```

## API配置

API配置在`config/api_config.yaml`文件中定义。以下是配置选项及其说明：

```yaml
# API基本配置
api:
  # API版本
  version: "v1"
  
  # 是否启用文档
  enable_docs: true
  
  # 文档URL
  docs_url: "/docs"
  
  # 是否启用CORS
  enable_cors: true
  
  # 允许的源
  allowed_origins: ["*"]
  
  # 是否启用速率限制
  enable_rate_limit: true

# 认证配置
auth:
  # 是否启用认证
  enable_auth: true
  
  # 认证类型，可选值为"token"、"jwt"或"basic"
  auth_type: "token"
  
  # 令牌过期时间（秒）
  token_expire: 86400
  
  # 是否启用刷新令牌
  enable_refresh_token: true

# 模型配置
model:
  # 默认模型路径
  default_model: "models/checkpoints/sample/final"
  
  # 可用模型列表
  available_models:
    - id: "model_1"
      name: "DeepSeek-1.5B-LoRA"
      path: "models/checkpoints/sample/final"
      description: "基于DeepSeek-R1 1.5B模型微调的通用问答模型"
    
    - id: "model_2"
      name: "DeepSeek-1.5B-医疗"
      path: "models/checkpoints/medical/final"
      description: "针对医疗领域微调的DeepSeek模型"

# 推理配置
inference:
  # 默认最大生成长度
  default_max_length: 100
  
  # 默认温度参数
  default_temperature: 0.7
  
  # 默认top-p参数
  default_top_p: 0.9
  
  # 默认top-k参数
  default_top_k: 50
  
  # 默认返回序列数量
  default_num_return_sequences: 1
  
  # 最大并发请求数
  max_concurrent_requests: 10
  
  # 请求超时时间（秒）
  request_timeout: 30
```

## 日志配置

日志配置在`config/logging_config.yaml`文件中定义。以下是配置选项及其说明：

```yaml
# 日志级别
level: INFO

# 日志格式
format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 日志处理器
handlers:
  # 控制台处理器
  console:
    enabled: true
    level: INFO
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  # 文件处理器
  file:
    enabled: true
    level: INFO
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    filename: "logs/app.log"
    max_bytes: 10485760  # 10MB
    backup_count: 5

# 第三方库日志级别
third_party:
  transformers: WARNING
  datasets: WARNING
  torch: WARNING
  accelerate: WARNING
```

## 配置文件加载顺序

项目按以下顺序加载配置：

1. 默认配置（硬编码在代码中）
2. 配置文件（YAML文件）
3. 环境变量（.env文件）
4. 命令行参数

后加载的配置会覆盖先加载的配置，因此命令行参数的优先级最高。

## 配置示例

### 训练配置示例

```yaml
model:
  base_model: "deepseek-ai/deepseek-llm-1.5b-base"
  max_length: 512
  quantization: "int8"

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  bias: "none"

training:
  batch_size: 8
  gradient_accumulation_steps: 2
  num_epochs: 3
  learning_rate: 3e-4
  fp16: true
  logging_steps: 100
  save_steps: 1000
  eval_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
  report_to: "tensorboard"
  weight_decay: 0.01
  warmup_steps: 500
  lr_scheduler_type: "cosine"

output_dir: "./models/checkpoints/sample"
```

### API配置示例

```yaml
api:
  version: "v1"
  enable_docs: true
  docs_url: "/docs"
  enable_cors: true
  allowed_origins: ["http://localhost:3000", "https://example.com"]
  enable_rate_limit: true

auth:
  enable_auth: true
  auth_type: "token"
  token_expire: 86400
  enable_refresh_token: true

model:
  default_model: "models/checkpoints/sample/final"
  available_models:
    - id: "model_1"
      name: "DeepSeek-1.5B-LoRA"
      path: "models/checkpoints/sample/final"
      description: "基于DeepSeek-R1 1.5B模型微调的通用问答模型"

inference:
  default_max_length: 100
  default_temperature: 0.7
  default_top_p: 0.9
  default_top_k: 50
  default_num_return_sequences: 1
  max_concurrent_requests: 10
  request_timeout: 30
```

## 配置最佳实践

1. **使用环境变量存储敏感信息**：API令牌、数据库密码等敏感信息应存储在环境变量中，而不是配置文件中。

2. **为不同环境使用不同配置**：为开发、测试和生产环境创建不同的配置文件，如`dev_config.yaml`、`test_config.yaml`和`prod_config.yaml`。

3. **使用合理的默认值**：为配置选项提供合理的默认值，以减少用户配置的复杂性。

4. **验证配置**：在应用启动时验证配置的有效性，确保所有必需的配置都已提供。

5. **记录配置变更**：记录配置的变更历史，以便追踪问题。

6. **使用配置模板**：提供配置模板，帮助用户快速创建自己的配置文件。 