import os
import torch
import logging
import json
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.config import MODEL_BASE_PATH, MODEL_DEVICE
from app.repositories.model_repository import ModelDAO

logger = logging.getLogger(__name__)

class ModelCoreOperations:
    """模型核心操作类，负责模型的加载、推理、数据处理和保存等核心功能"""
    def __init__(self):
        """初始化模型核心操作类，设置模型、分词器和设备"""
        self.model = None  # 语言模型实例
        self.tokenizer = None  # 分词器实例
        self.device = MODEL_DEVICE  # 模型运行的设备（CPU/GPU）
        
    def load_model(self, model_path):
        """Load model from specified path"""
        try:
            logger.info(f'Loading model from {model_path}')
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.to(self.device)
            logger.info('Model loaded successfully')
            return True
        except Exception as e:
            logger.error(f'Failed to load model: {str(e)}')
            raise

    def generate(self, prompt, max_length=100, temperature=0.7):
        """Generate text from prompt"""
        if not self.model or not self.tokenizer:
            raise ValueError('Model not loaded')
            
        if not prompt or not isinstance(prompt, str):
            raise ValueError('Prompt must be a non-empty string')
            
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError('max_length must be a positive integer')
            
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError('temperature must be a positive number')
            
        try:
            logger.info(f'Generating text with max_length={max_length}, temperature={temperature}')
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f'Generated text: {generated_text[:100]}...')
            
            return generated_text
        except Exception as e:
            logger.error(f'Text generation failed: {str(e)}')
            raise ValueError('Failed to generate text') from e

    def _load_dataset(self, data_path):
        """Load and preprocess dataset with enhanced caching and parallel processing"""
        try:
            logger.info(f'Loading dataset from {data_path}')
            
            # Validate input path and file type
            if not os.path.exists(data_path):
                raise FileNotFoundError(f'Data path {data_path} does not exist')
            if not os.path.isfile(data_path):
                raise ValueError(f'Data path {data_path} is not a file')
                
            # Create cache directory with proper permissions
            cache_dir = os.path.join(MODEL_BASE_PATH, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            os.chmod(cache_dir, 0o755)  # Ensure proper permissions
            
            # Generate robust cache key with additional metadata
            import hashlib
            cache_key = hashlib.sha256()
            
            # Include file content in cache key
            with open(data_path, 'rb') as f:
                while chunk := f.read(8192):
                    cache_key.update(chunk)
                    
            # Include model config and tokenizer info
            cache_key.update(str(self.model.config).encode())
            cache_key.update(str(self.tokenizer.get_vocab()).encode())
            
            # Include system and environment info
            import platform
            import torch
            cache_key.update(platform.platform().encode())
            cache_key.update(torch.__version__.encode())
            cache_key.update(str(torch.cuda.is_available()).encode())
            
            cache_path = os.path.join(
                cache_dir,
                f'cached_{cache_key.hexdigest()}.pt'
            )
            
            # Add cache metadata
            cache_metadata = {
                'data_path': data_path,
                'model_config': str(self.model.config),
                'tokenizer_vocab': str(self.tokenizer.get_vocab()),
                'platform': platform.platform(),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'created_at': datetime.datetime.now().isoformat()
            }
            
            # Check cache validity
            if os.path.exists(cache_path):
                cache_mtime = os.path.getmtime(cache_path)
                data_mtime = os.path.getmtime(data_path)
                
                if cache_mtime > data_mtime:
                    logger.info('Loading cached preprocessed data')
                    try:
                        # Load with memory mapping and validation
                        cached_data = torch.load(cache_path, map_location='cpu', mmap=True)
                        
                        # Validate cached data structure
                        if not isinstance(cached_data, list):
                            raise ValueError('Invalid cached data format')
                        if len(cached_data) == 0:
                            raise ValueError('Empty cached data')
                        if not all(isinstance(item, dict) for item in cached_data):
                            raise ValueError('Invalid cached data items')
                            
                        logger.info(f'Loaded {len(cached_data)} validated samples from cache')
                        return cached_data
                    except Exception as e:
                        logger.warning(f'Cache validation failed: {str(e)}')
                        # Fall through to reprocess data
                
            # Process data based on format
            processed_data = None
            if data_path.endswith('.jsonl'):
                # Load JSON Lines format with error handling
                import json
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f'Invalid JSON at line {line_num}: {str(e)}')
                            continue
                processed_data = self._preprocess_data(data)
                
            elif data_path.endswith('.csv'):
                # Load CSV format with chunking
                import pandas as pd
                try:
                    # Read in chunks to handle large files
                    chunks = pd.read_csv(data_path, chunksize=10000)
                    data = []
                    for chunk in chunks:
                        data.extend(chunk.to_dict('records'))
                    processed_data = self._preprocess_data(data)
                except Exception as e:
                    raise ValueError(f'Failed to read CSV: {str(e)}')
                
            elif data_path.endswith('.txt'):
                # Load plain text format with encoding validation
                try:
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                    processed_data = self._preprocess_text_data(data)
                except UnicodeDecodeError:
                    # Try alternative encodings
                    try:
                        with open(data_path, 'r', encoding='latin-1') as f:
                            data = f.read()
                        processed_data = self._preprocess_text_data(data)
                    except Exception as e:
                        raise ValueError(f'Failed to decode text file: {str(e)}')
                
            else:
                raise ValueError(f'Unsupported file format: {data_path}')
            
            # Save processed data to cache
            if processed_data:
                try:
                    torch.save(processed_data, cache_path)
                    os.chmod(cache_path, 0o644)  # Set proper permissions
                    logger.info(f'Saved processed data to cache: {cache_path}')
                except Exception as e:
                    logger.warning(f'Failed to save cache: {str(e)}')
                    
            return processed_data
            
        except Exception as e:
            logger.error(f'Failed to load dataset: {str(e)}')
            raise

    def _preprocess_data(self, data):
        """
        预处理结构化数据（JSONL/CSV），使用批量处理
        
        参数:
            data: 包含文本数据的结构化数据列表
            
        返回:
            处理后的数据列表，包含input_ids、attention_mask和labels
            
        处理流程:
        1. 验证数据格式，确保每个数据项都包含text字段
        2. 将文本数据分批处理，避免内存问题
        3. 使用分词器对文本进行动态填充和截断
        4. 对部分数据进行随机掩码处理，用于数据增强
        """
        if not self.tokenizer:
            raise ValueError('Tokenizer not loaded')
            
        # Validate data
        if not all('text' in item for item in data):
            raise ValueError('All data items must contain "text" field')
            
        # Batch tokenization with dynamic padding
        texts = [item['text'] for item in data]
        batch_size = 1000  # Process in chunks to avoid memory issues
        processed_data = []
        
        # Configure tokenizer for better performance
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch with dynamic padding
            tokenized = self.tokenizer(
                batch_texts,
                padding=True,  # Dynamic padding
                truncation=True,
                max_length=512,
                return_tensors='pt',
                return_attention_mask=True,
                return_special_tokens_mask=True
            )
            
            # Convert to list of dicts with data augmentation
            for j in range(len(batch_texts)):
                # Apply random masking for data augmentation
                input_ids = tokenized['input_ids'][j]
                special_tokens_mask = tokenized['special_tokens_mask'][j]
                
                # Create masked version with 15% probability
                if torch.rand(1).item() < 0.15:
                    input_ids = self._apply_random_masking(
                        input_ids,
                        special_tokens_mask
                    )
                
                processed_data.append({
                    'input_ids': input_ids,
                    'attention_mask': tokenized['attention_mask'][j],
                    'labels': tokenized['input_ids'][j].clone()
                })
            
        return processed_data

    def _preprocess_text_data(self, text):
        """
        预处理纯文本数据
        
        参数:
            text: 原始文本字符串
            
        返回:
            处理后的数据列表，包含input_ids、attention_mask和labels
            
        处理流程:
        1. 使用分词器将文本编码为token序列
        2. 将token序列切分为固定长度的chunk
        3. 对每个chunk进行填充和截断处理
        4. 返回处理后的数据列表
        """
        if not self.tokenizer:
            raise ValueError('Tokenizer not loaded')
            
        # Split text into chunks
        chunk_size = 512
        tokens = self.tokenizer.encode(text)
        chunks = [
            tokens[i:i + chunk_size]
            for i in range(0, len(tokens), chunk_size)
        ]
        
        processed_data = []
        for chunk in chunks:
            tokenized = self.tokenizer.prepare_for_model(
                chunk,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            processed_data.append({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': tokenized['input_ids'].clone()
            })
            
        return processed_data

    def _create_data_loader(self, dataset, batch_size):
        """
        创建优化的数据加载器，支持增强的并行处理
        
        参数:
            dataset: 预处理后的数据集
            batch_size: 每个batch的大小
            
        返回:
            带有统计跟踪功能的数据加载器
            
        功能特点:
        1. 使用自定义数据集类实现缓存优化
        2. 根据系统资源动态计算最优的worker数量
        3. 支持内存映射和持久化worker
        4. 包含内存使用监控和统计功能
        """
        try:
            logger.info(f'Creating data loader with batch size {batch_size}')
            
            # Create custom dataset class with enhanced caching
            class TextDataset(torch.utils.data.Dataset):
                def __init__(self, data):
                    self.data = data
                    self.cache = {}
                    self.lock = threading.Lock()  # Thread-safe cache access
                    
                def __len__(self):
                    return len(self.data)
                    
                def __getitem__(self, idx):
                    # Use cached item if available
                    with self.lock:
                        if idx in self.cache:
                            return self.cache[idx]
                            
                    # Process data
                    item = {
                        'input_ids': self.data[idx]['input_ids'].squeeze(0),
                        'attention_mask': self.data[idx]['attention_mask'].squeeze(0),
                        'labels': self.data[idx]['labels'].squeeze(0)
                    }
                    
                    # Cache item with thread-safe access
                    with self.lock:
                        self.cache[idx] = item
                    return item
                    
                def clear_cache(self):
                    """Clear cache to free memory"""
                    with self.lock:
                        self.cache.clear()
            
            # Create dataset instance
            text_dataset = TextDataset(dataset)
            
            # Calculate optimal number of workers based on system resources
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            # Dynamic worker calculation based on system resources
            num_workers = min(
                max(2, cpu_count - 1),  # At least 2 workers
                max(4, gpu_count * 4)   # 4 workers per GPU
            )
            
            # Configure DataLoader with enhanced parallel settings
            data_loader = torch.utils.data.DataLoader(
                text_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),  # Only pin memory if GPU available
                persistent_workers=True,
                prefetch_factor=4 if num_workers > 0 else None,  # Increased prefetch
                worker_init_fn=lambda worker_id: torch.manual_seed(
                    torch.initial_seed() + worker_id
                )  # Ensure reproducibility
            )
            
            # Add memory monitoring
            class TrackedDataLoader:
                def __init__(self, data_loader):
                    self.data_loader = data_loader
                    self.batches_processed = 0
                    self.total_samples = 0
                    self.memory_usage = []
                    
                def __iter__(self):
                    self.batches_processed = 0
                    self.total_samples = 0
                    self.memory_usage = []
                    return self
                    
                def __next__(self):
                    # Track memory before batch processing
                    process = psutil.Process(os.getpid())
                    mem_before = process.memory_info().rss
                    
                    batch = next(self.data_loader)
                    
                    # Track memory after batch processing
                    mem_after = process.memory_info().rss
                    self.memory_usage.append((mem_before, mem_after))
                    
                    self.batches_processed += 1
                    self.total_samples += batch['input_ids'].size(0)
                    return batch
                    
                def __len__(self):
                    return len(self.data_loader)
                    
                def get_stats(self):
                    return {
                        'batches_processed': self.batches_processed,
                        'total_samples': self.total_samples,
                        'memory_usage': self.memory_usage
                    }
            
            # Log data loader configuration
            logger.info(f'Created data loader with:')
            logger.info(f'- Workers: {num_workers}')
            logger.info(f'- Prefetch factor: 2')
            logger.info(f'- Pin memory: True')
            logger.info(f'- Persistent workers: True')
            
            # Add data loader statistics tracking
            class TrackedDataLoader:
                def __init__(self, data_loader):
                    self.data_loader = data_loader
                    self.batches_processed = 0
                    self.total_samples = 0
                    
                def __iter__(self):
                    self.batches_processed = 0
                    self.total_samples = 0
                    return self
                    
                def __next__(self):
                    batch = next(self.data_loader)
                    self.batches_processed += 1
                    self.total_samples += batch['input_ids'].size(0)
                    return batch
                    
                def __len__(self):
                    return len(self.data_loader)
                    
                def get_stats(self):
                    return {
                        'batches_processed': self.batches_processed,
                        'total_samples': self.total_samples
                    }
            
            return TrackedDataLoader(data_loader)
            
        except Exception as e:
            logger.error(f'Failed to create data loader: {str(e)}')
            raise

    def save_model(self, version, description=None):
        """Save current model as new version with enhanced metadata"""
        if not self.model or not self.tokenizer:
            raise ValueError('Model not loaded')
            
        save_path = os.path.join(MODEL_BASE_PATH, version)
        
        # Create directory if not exists
        os.makedirs(save_path, exist_ok=True)
        
        # Generate model metadata
        model_metadata = {
            'version': version,
            'description': description,
            'model_type': self.model.config.model_type,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'architecture': str(self.model.config.architectures),
            'created_at': datetime.datetime.now().isoformat(),
            'training_config': getattr(self, 'training_config', None)
        }
        
        # Save model and tokenizer with additional metadata
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save metadata
        with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Validate saved model
        self._validate_saved_model(save_path)
        
        # Create model version using DAO
        model_version = ModelDAO.create_model_version(
            version,
            save_path,
            description,
            metadata=model_metadata
        )
        
        logger.info(f'Saved model version {version} with {model_metadata["num_parameters"]} parameters')
        return model_version

    def _validate_saved_model(self, model_path):
        """
        验证保存的模型完整性
        
        参数:
            model_path: 模型保存路径
            
        返回:
            验证是否成功
            
        验证步骤:
        1. 检查模型必需文件是否存在
        2. 加载临时模型和分词器
        3. 执行前向传播测试
        4. 记录验证结果
        """
        try:
            logger.info(f'Validating saved model at {model_path}')
            
            # Check required files exist
            required_files = [
                'pytorch_model.bin',
                'config.json',
                'tokenizer.json',
                'vocab.json',
                'special_tokens_map.json'
            ]
            
            for file in required_files:
                if not os.path.exists(os.path.join(model_path, file)):
                    raise ValueError(f'Missing required file: {file}')
            
            # Verify model can be loaded
            temp_model = AutoModelForCausalLM.from_pretrained(model_path)
            temp_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Test forward pass
            test_input = temp_tokenizer("Test input", return_tensors="pt")
            _ = temp_model(**test_input)
            
            logger.info('Model validation successful')
            return True
        except Exception as e:
            logger.error(f'Model validation failed: {str(e)}')

    def _apply_random_masking(self, input_ids, special_tokens_mask):
        """
        应用随机掩码和其他数据增强技术
        
        参数:
            input_ids: 输入token的ID序列
            special_tokens_mask: 特殊token的掩码
            
        返回:
            应用随机掩码后的input_ids
            
        实现细节:
        1. 创建非特殊token的掩码
        2. 随机选择15%的非特殊token进行掩码
        3. 返回处理后的input_ids
        """
        # Create mask for non-special tokens
        mask = (special_tokens_mask == 0)
        mask_indices = torch.nonzero(mask).squeeze()
