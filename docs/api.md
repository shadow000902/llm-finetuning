# API 文档

本文档详细说明了项目提供的RESTful API接口，包括认证方式、请求示例、响应格式和错误代码等。

## 基本信息

- 基础URL: `http://localhost:5000/api/v1`
- 内容类型: `application/json`
- 认证方式: Bearer Token

## 认证

所有API请求都需要在HTTP头部包含认证令牌：

```
Authorization: Bearer YOUR_TOKEN
```

认证令牌可以通过登录API获取。

### 登录获取令牌

```
POST /login
```

**请求参数**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| username | string | 是 | 用户名 |
| password | string | 是 | 密码 |

**响应**

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

## API端点

### 1. 模型训练

#### 启动训练任务

```
POST /train
```

启动一个新的模型训练任务。

**请求参数**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| dataset_path | string | 是 | 训练数据集路径 |
| config | object | 否 | 训练配置参数 |
| config.learning_rate | number | 否 | 学习率，默认为3e-5 |
| config.batch_size | integer | 否 | 批次大小，默认为8 |
| config.num_epochs | integer | 否 | 训练轮数，默认为3 |
| config.output_dir | string | 否 | 输出目录，默认为"./models/checkpoints" |
| config.quantization | string | 否 | 量化类型，可选值为"int8"、"int4"或null |
| lora_config | object | 否 | LoRA配置参数 |
| lora_config.r | integer | 否 | LoRA秩，默认为8 |
| lora_config.lora_alpha | integer | 否 | LoRA alpha参数，默认为16 |
| lora_config.lora_dropout | number | 否 | LoRA dropout率，默认为0.05 |

**请求示例**

```json
{
  "dataset_path": "/data/processed/sample",
  "config": {
    "learning_rate": 3e-5,
    "batch_size": 16,
    "num_epochs": 3,
    "output_dir": "./models/my_model"
  },
  "lora_config": {
    "r": 16,
    "lora_alpha": 32
  }
}
```

**响应**

```json
{
  "task_id": "12345678-1234-5678-1234-567812345678",
  "status": "started",
  "message": "训练任务已启动",
  "created_at": "2023-03-08T12:00:00Z"
}
```

#### 获取训练状态

```
GET /train/status/{task_id}
```

获取指定训练任务的状态。

**路径参数**

| 参数 | 类型 | 描述 |
|------|------|------|
| task_id | string | 训练任务ID |

**响应**

```json
{
  "task_id": "12345678-1234-5678-1234-567812345678",
  "status": "running",
  "progress": 45.5,
  "metrics": {
    "train_loss": 2.1543,
    "learning_rate": 2.8e-5,
    "epoch": 1.5
  },
  "started_at": "2023-03-08T12:00:00Z",
  "updated_at": "2023-03-08T12:30:00Z"
}
```

**状态值说明**

- `queued`: 任务已加入队列，等待执行
- `running`: 任务正在执行
- `completed`: 任务已完成
- `failed`: 任务执行失败
- `canceled`: 任务已取消

#### 取消训练任务

```
POST /train/cancel/{task_id}
```

取消指定的训练任务。

**路径参数**

| 参数 | 类型 | 描述 |
|------|------|------|
| task_id | string | 训练任务ID |

**响应**

```json
{
  "task_id": "12345678-1234-5678-1234-567812345678",
  "status": "canceled",
  "message": "训练任务已取消",
  "canceled_at": "2023-03-08T13:00:00Z"
}
```

### 2. 模型推理

#### 生成文本

```
POST /generate
```

使用模型生成文本。

**请求参数**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| prompt | string | 是 | 输入提示文本 |
| model_path | string | 否 | 模型路径，默认使用最新训练的模型 |
| max_length | integer | 否 | 最大生成长度，默认为100 |
| temperature | number | 否 | 温度参数，默认为0.7 |
| top_p | number | 否 | 核采样参数，默认为0.9 |
| top_k | integer | 否 | top-k采样参数，默认为50 |
| num_return_sequences | integer | 否 | 返回序列数量，默认为1 |

**请求示例**

```json
{
  "prompt": "请解释量子计算的基本原理",
  "max_length": 200,
  "temperature": 0.8,
  "top_p": 0.95,
  "num_return_sequences": 1
}
```

**响应**

```json
{
  "prompt": "请解释量子计算的基本原理",
  "generated_texts": [
    "量子计算是一种利用量子力学原理进行信息处理的计算方式。与经典计算使用比特（0或1）不同，量子计算使用量子比特（qubit），它可以同时处于多个状态的叠加。这种特性使得量子计算机在处理特定问题时，如大数分解、搜索和模拟量子系统等，可能比经典计算机快得多。"
  ],
  "model_path": "models/checkpoints/sample/final",
  "generation_time": 1.25,
  "timestamp": "2023-03-08T14:00:00Z"
}
```

#### 批量生成文本

```
POST /generate/batch
```

批量生成文本。

**请求参数**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| prompts | array | 是 | 输入提示文本数组 |
| model_path | string | 否 | 模型路径，默认使用最新训练的模型 |
| max_length | integer | 否 | 最大生成长度，默认为100 |
| temperature | number | 否 | 温度参数，默认为0.7 |
| top_p | number | 否 | 核采样参数，默认为0.9 |
| top_k | integer | 否 | top-k采样参数，默认为50 |
| num_return_sequences | integer | 否 | 每个提示返回的序列数量，默认为1 |

**请求示例**

```json
{
  "prompts": [
    "请解释量子计算的基本原理",
    "简述人工智能在医疗领域的应用"
  ],
  "max_length": 200,
  "temperature": 0.8
}
```

**响应**

```json
{
  "results": [
    {
      "prompt": "请解释量子计算的基本原理",
      "generated_texts": [
        "量子计算是一种利用量子力学原理进行信息处理的计算方式。与经典计算使用比特（0或1）不同，量子计算使用量子比特（qubit），它可以同时处于多个状态的叠加。这种特性使得量子计算机在处理特定问题时，如大数分解、搜索和模拟量子系统等，可能比经典计算机快得多。"
      ]
    },
    {
      "prompt": "简述人工智能在医疗领域的应用",
      "generated_texts": [
        "人工智能在医疗领域有广泛的应用，主要包括：1）医学影像分析：AI可以帮助分析X光片、CT扫描和MRI图像，提高疾病诊断的准确性和速度；2）药物研发：AI可以加速新药发现过程，预测药物与蛋白质的相互作用，并优化分子设计；3）个性化医疗：通过分析患者的基因组数据和医疗历史，AI可以帮助制定个性化治疗方案。"
      ]
    }
  ],
  "model_path": "models/checkpoints/sample/final",
  "total_generation_time": 2.5,
  "timestamp": "2023-03-08T14:05:00Z"
}
```

### 3. 模型管理

#### 获取可用模型列表

```
GET /models
```

获取系统中可用的模型列表。

**响应**

```json
{
  "models": [
    {
      "id": "model_1",
      "name": "DeepSeek-1.5B-LoRA",
      "path": "models/checkpoints/sample/final",
      "base_model": "deepseek-ai/deepseek-llm-1.5b-base",
      "created_at": "2023-03-07T10:00:00Z",
      "size_mb": 256,
      "description": "基于DeepSeek-R1 1.5B模型微调的通用问答模型"
    },
    {
      "id": "model_2",
      "name": "DeepSeek-1.5B-医疗",
      "path": "models/checkpoints/medical/final",
      "base_model": "deepseek-ai/deepseek-llm-1.5b-base",
      "created_at": "2023-03-08T09:30:00Z",
      "size_mb": 260,
      "description": "针对医疗领域微调的DeepSeek模型"
    }
  ],
  "count": 2,
  "timestamp": "2023-03-08T15:00:00Z"
}
```

#### 获取模型详情

```
GET /models/{model_id}
```

获取指定模型的详细信息。

**路径参数**

| 参数 | 类型 | 描述 |
|------|------|------|
| model_id | string | 模型ID |

**响应**

```json
{
  "id": "model_1",
  "name": "DeepSeek-1.5B-LoRA",
  "path": "models/checkpoints/sample/final",
  "base_model": "deepseek-ai/deepseek-llm-1.5b-base",
  "created_at": "2023-03-07T10:00:00Z",
  "size_mb": 256,
  "description": "基于DeepSeek-R1 1.5B模型微调的通用问答模型",
  "training_config": {
    "learning_rate": 3e-5,
    "batch_size": 16,
    "num_epochs": 3,
    "lora_r": 8,
    "lora_alpha": 16
  },
  "metrics": {
    "perplexity": 8.45,
    "train_loss": 1.85,
    "val_loss": 2.12
  },
  "last_used": "2023-03-08T14:30:00Z",
  "usage_count": 42
}
```

### 4. 数据集管理

#### 获取数据集列表

```
GET /datasets
```

获取系统中可用的数据集列表。

**响应**

```json
{
  "datasets": [
    {
      "id": "dataset_1",
      "name": "通用指令数据集",
      "path": "data/processed/sample",
      "created_at": "2023-03-06T09:00:00Z",
      "size_mb": 15,
      "samples_count": 10000,
      "description": "通用指令问答数据集"
    },
    {
      "id": "dataset_2",
      "name": "医疗领域数据集",
      "path": "data/processed/medical",
      "created_at": "2023-03-07T11:20:00Z",
      "size_mb": 8,
      "samples_count": 5000,
      "description": "医疗领域专业问答数据集"
    }
  ],
  "count": 2,
  "timestamp": "2023-03-08T15:10:00Z"
}
```

#### 上传数据集

```
POST /datasets/upload
```

上传新的数据集。

**请求参数**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| file | file | 是 | 数据集文件（JSON或CSV格式） |
| name | string | 是 | 数据集名称 |
| description | string | 否 | 数据集描述 |
| format | string | 否 | 数据格式，可选值为"json"或"csv"，默认为"json" |

**响应**

```json
{
  "id": "dataset_3",
  "name": "金融领域数据集",
  "path": "data/raw/finance.json",
  "uploaded_at": "2023-03-08T15:20:00Z",
  "size_mb": 12,
  "samples_count": 8000,
  "description": "金融领域专业问答数据集",
  "message": "数据集上传成功"
}
```

#### 处理数据集

```
POST /datasets/process
```

处理原始数据集，转换为模型训练所需的格式。

**请求参数**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| dataset_id | string | 是 | 数据集ID |
| output_name | string | 是 | 输出数据集名称 |
| train_ratio | number | 否 | 训练集比例，默认为0.8 |
| val_ratio | number | 否 | 验证集比例，默认为0.1 |
| test_ratio | number | 否 | 测试集比例，默认为0.1 |
| template | string | 否 | 提示模板 |

**请求示例**

```json
{
  "dataset_id": "dataset_3",
  "output_name": "finance_processed",
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "template": "### 指令:\n{instruction}\n\n### 回答:\n{response}"
}
```

**响应**

```json
{
  "task_id": "87654321-8765-4321-8765-432187654321",
  "status": "completed",
  "dataset_id": "dataset_3",
  "output_path": "data/processed/finance_processed",
  "splits": {
    "train": 6400,
    "validation": 800,
    "test": 800
  },
  "message": "数据集处理完成",
  "processed_at": "2023-03-08T15:25:00Z"
}
```

## 错误代码

API可能返回以下错误代码：

| 状态码 | 错误代码 | 描述 |
|--------|----------|------|
| 400 | INVALID_REQUEST | 请求参数无效 |
| 401 | UNAUTHORIZED | 未授权访问 |
| 403 | FORBIDDEN | 禁止访问 |
| 404 | NOT_FOUND | 资源不存在 |
| 409 | CONFLICT | 资源冲突 |
| 429 | TOO_MANY_REQUESTS | 请求过于频繁 |
| 500 | INTERNAL_ERROR | 服务器内部错误 |
| 503 | SERVICE_UNAVAILABLE | 服务不可用 |

**错误响应示例**

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "参数'prompt'不能为空",
    "details": {
      "field": "prompt",
      "reason": "required"
    }
  },
  "timestamp": "2023-03-08T16:00:00Z",
  "request_id": "req-12345678"
}
```

## 速率限制

API实施了速率限制，以防止过度使用：

- 训练API: 每小时10次请求
- 推理API: 每分钟60次请求
- 其他API: 每分钟120次请求

超过限制时，API将返回429状态码。响应头部包含以下信息：

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1678291200
```

## 版本控制

当前API版本为v1。API版本包含在URL路径中：`/api/v1/`。

未来版本将使用新的路径，如`/api/v2/`，以确保向后兼容性。 