# 项目使用说明

本文档提供了项目的详细使用说明，包括环境配置、数据处理、模型训练、推理和评估等步骤。

## 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/shadow000902/llm-finetuning.git
cd llm-finetuning
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制.env.example文件为.env，并根据实际情况修改配置：

```bash
cp .env.example .env
```

## 数据准备

### 1. 准备训练数据

训练数据应为JSON格式，包含指令和回答：

```json
[
  {
    "instruction": "解释量子计算的基本原理",
    "response": "量子计算是一种利用量子力学原理进行信息处理的计算方式..."
  },
  ...
]
```

### 2. 处理数据

使用数据处理脚本将原始数据转换为模型训练所需的格式：

```bash
python -m app.scripts.process_data \
  --input data/raw/sample_data.json \
  --output data/processed/sample \
  --format json \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

更多参数说明请参考 [脚本使用说明](app/scripts/README.md#数据处理)。

## 模型训练

### 1. 配置训练参数

修改 `config/train_config.yaml` 文件，设置训练参数：

```yaml
model:
  base_model: "deepseek-ai/deepseek-llm-1.5b-base"
  max_length: 512
  quantization: "int8"  # 可选: "int4", null

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  bias: "none"

training:
  batch_size: 8
  gradient_accumulation_steps: 1
  num_epochs: 3
  learning_rate: 3e-4
  fp16: true
  logging_steps: 100
  save_steps: 1000
  eval_steps: 500
  save_total_limit: 3
  load_best_model_at_end: true
  report_to: "tensorboard"

output_dir: "./models/checkpoints/sample"
```

### 2. 启动训练

使用训练脚本启动模型训练：

```bash
python -m app.scripts.train_model \
  --config config/train_config.yaml \
  --data-dir data/processed/sample \
  --output-dir models/checkpoints/sample \
  --log-file logs/training.log \
  --log-level INFO
```

更多参数说明请参考 [脚本使用说明](app/scripts/README.md#模型训练)。

### 3. 监控训练

训练过程中可以使用TensorBoard监控训练进度：

```bash
tensorboard --logdir models/checkpoints/sample
```

## 模型推理

### 1. 准备提示文本

创建提示文本文件，如 `data/prompts/sample_prompts.json`：

```json
[
  "请解释量子计算的基本原理",
  "简述人工智能在医疗领域的应用",
  ...
]
```

### 2. 运行推理

使用推理脚本进行文本生成：

```bash
python -m app.scripts.inference \
  --model-path models/checkpoints/sample/final \
  --prompts data/prompts/sample_prompts.json \
  --output results/inference_results.json \
  --max-length 512 \
  --temperature 0.7 \
  --log-file logs/inference.log
```

更多参数说明请参考 [脚本使用说明](app/scripts/README.md#模型推理)。

## 模型评估

使用评估脚本评估模型性能：

```bash
python -m app.scripts.evaluate_model \
  --model-path models/checkpoints/sample/final \
  --data-dir data/processed/sample \
  --output results/evaluation_result.json \
  --max-samples 100 \
  --log-file logs/evaluation.log
```

更多参数说明请参考 [脚本使用说明](app/scripts/README.md#模型评估)。

## API服务

### 1. 启动API服务

```bash
python run.py
```

### 2. API使用示例

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

## 常见问题

### 1. 训练过程中出现OOM错误怎么办？

可以尝试以下方法：
- 减小batch size
- 启用梯度检查点（gradient checkpointing）
- 使用混合精度训练
- 使用量化训练（int8或int4）

### 2. 如何选择合适的batch size？

建议从16开始，根据GPU显存逐步增加，直到达到显存上限的80%。

### 3. 如何评估模型效果？

可以使用以下指标：
- 困惑度（Perplexity）
- BLEU分数
- 人工评估

### 4. 如何部署模型到生产环境？

可以使用以下方法：
- 使用Docker容器化部署
- 使用云服务提供商的托管服务
- 使用Kubernetes进行容器编排

## 更多资源

- [项目文档](README.md)
- [API文档](docs/api.md)
- [脚本使用说明](app/scripts/README.md)
- [配置说明](docs/configuration.md) 