# 脚本使用说明

本目录包含了用于数据处理、模型训练、推理和评估的脚本。

## 数据处理

`process_data.py` 脚本用于将原始数据转换为模型训练所需的格式。

### 使用示例

```bash
# 处理JSON格式的数据
python -m app.scripts.process_data \
  --input data/raw/sample_data.json \
  --output data/processed/sample \
  --format json \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1

# 处理CSV格式的数据
python -m app.scripts.process_data \
  --input data/raw/sample_data.csv \
  --output data/processed/sample \
  --format csv \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1

# 使用自定义模板
python -m app.scripts.process_data \
  --input data/raw/sample_data.json \
  --output data/processed/sample \
  --format json \
  --template "问题：{instruction}\n\n答案：{response}"
```

### 参数说明

- `--input`: 输入数据文件路径
- `--output`: 输出目录
- `--format`: 输入数据格式，可选值为 `json` 或 `csv`
- `--template`: 提示模板，默认为 `### 指令:\n{instruction}\n\n### 回答:\n{response}`
- `--train-ratio`: 训练集比例，默认为 0.8
- `--val-ratio`: 验证集比例，默认为 0.1
- `--test-ratio`: 测试集比例，默认为 0.1
- `--seed`: 随机种子，默认为 42

## 模型训练

`train_model.py` 脚本用于启动模型训练。

### 使用示例

```bash
# 使用配置文件训练模型
python -m app.scripts.train_model \
  --config config/train_config.yaml \
  --data-dir data/processed/sample \
  --output-dir models/checkpoints/sample \
  --log-file logs/training.log \
  --log-level INFO
```

### 参数说明

- `--config`: 配置文件路径
- `--data-dir`: 数据目录
- `--output-dir`: 输出目录
- `--log-file`: 日志文件路径
- `--log-level`: 日志级别，可选值为 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`

## 模型推理

`inference.py` 脚本用于使用训练好的模型进行文本生成。

### 使用示例

```bash
# 使用单个提示文本
python -m app.scripts.inference \
  --model-path models/checkpoints/sample/final \
  --prompt "请解释量子计算的基本原理" \
  --output results/inference_result.json \
  --max-length 512 \
  --temperature 0.7 \
  --log-file logs/inference.log

# 使用提示文本文件
python -m app.scripts.inference \
  --model-path models/checkpoints/sample/final \
  --prompts data/prompts/sample_prompts.json \
  --output results/inference_results.json \
  --max-length 512 \
  --temperature 0.7 \
  --log-file logs/inference.log

# 使用量化模型
python -m app.scripts.inference \
  --model-path models/checkpoints/sample/final \
  --prompts data/prompts/sample_prompts.json \
  --output results/inference_results.json \
  --quantization int8 \
  --log-file logs/inference.log
```

### 参数说明

- `--model-path`: 模型路径
- `--prompts`: 提示文本文件路径
- `--prompt`: 单个提示文本
- `--output`: 输出文件路径
- `--max-length`: 最大生成长度，默认为 512
- `--temperature`: 温度参数，默认为 0.7
- `--top-p`: 核采样参数，默认为 0.9
- `--top-k`: top-k采样参数，默认为 50
- `--num-return-sequences`: 返回序列数量，默认为 1
- `--quantization`: 量化类型，可选值为 `int8` 或 `int4`
- `--log-file`: 日志文件路径
- `--log-level`: 日志级别，可选值为 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`

## 模型评估

`evaluate_model.py` 脚本用于评估模型性能。

### 使用示例

```bash
# 评估模型性能
python -m app.scripts.evaluate_model \
  --model-path models/checkpoints/sample/final \
  --data-dir data/processed/sample \
  --output results/evaluation_result.json \
  --max-samples 100 \
  --log-file logs/evaluation.log

# 使用量化模型评估
python -m app.scripts.evaluate_model \
  --model-path models/checkpoints/sample/final \
  --data-dir data/processed/sample \
  --output results/evaluation_result.json \
  --quantization int8 \
  --log-file logs/evaluation.log
```

### 参数说明

- `--model-path`: 模型路径
- `--data-dir`: 数据目录
- `--output`: 输出文件路径
- `--config`: 配置文件路径
- `--max-samples`: 最大评估样本数，默认为 100
- `--quantization`: 量化类型，可选值为 `int8` 或 `int4`
- `--log-file`: 日志文件路径
- `--log-level`: 日志级别，可选值为 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`
