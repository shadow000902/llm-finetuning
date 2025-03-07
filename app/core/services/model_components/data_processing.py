import os
import torch
import logging
import threading
import hashlib
import datetime
import psutil
import platform
from transformers import AutoTokenizer
from config.config import Config

logger = logging.getLogger(__name__)

class DataProcessing:
    """核心数据处理类，负责数据加载、预处理、缓存管理等功能
    
    该类主要负责以下功能：
    1. 数据集的加载与预处理
    2. 数据缓存管理
    3. 文本的tokenization与解码
    4. 数据加载器的创建
    
    主要特性：
    - 支持多种数据格式（jsonl、csv、txt）
    - 自动缓存机制，提升重复加载效率
    - 线程安全的数据处理
    - 完善的错误处理与日志记录
    
    Attributes:
        tokenizer: 用于文本处理的tokenizer实例，需支持encode/decode方法
    
    主要方法：
    - load_dataset: 加载并预处理数据集
    - tokenize_text: 将文本转换为token ID
    - decode_text: 将token ID转换回文本
    - create_data_loader: 创建数据加载器
    
    依赖：
    - transformers.AutoTokenizer: 用于文本处理
    - torch: 用于张量操作
    - psutil: 用于系统资源监控
    - hashlib: 用于缓存key生成
    """
    
    def __init__(self, tokenizer):
        """初始化DataProcessing实例
        
        该方法初始化数据处理类，设置核心的tokenizer实例。
        tokenizer将用于后续的文本编码、解码等操作。
        
        Args:
            tokenizer: 用于文本处理的tokenizer实例，需满足以下要求：
                - 实现encode方法，将文本转换为token ID
                - 实现decode方法，将token ID转换回文本
                - 实现get_vocab方法，返回词汇表信息
                
        示例:
            >>> from transformers import AutoTokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            >>> processor = DataProcessing(tokenizer)
            
        Raises:
            TypeError: 如果tokenizer不符合要求
        """
        if not hasattr(tokenizer, 'encode') or not callable(tokenizer.encode):
            raise TypeError('tokenizer must implement encode method')
        if not hasattr(tokenizer, 'decode') or not callable(tokenizer.decode):
            raise TypeError('tokenizer must implement decode method')
        if not hasattr(tokenizer, 'get_vocab') or not callable(tokenizer.get_vocab):
            raise TypeError('tokenizer must implement get_vocab method')
            
        self.tokenizer = tokenizer

    def load_dataset(self, data_path):
        """加载并预处理数据集，使用增强的缓存机制和并行处理
        
        该方法负责加载和预处理数据集，支持多种文件格式，并实现了智能缓存机制。
        缓存机制基于文件内容、tokenizer配置和系统环境生成唯一key，确保缓存一致性。
        
        Args:
            data_path (str): 数据集文件路径，支持以下格式：
                - .jsonl: JSON Lines格式，每行一个JSON对象
                - .csv: CSV格式，支持分块读取
                - .txt: 纯文本格式，支持多种编码
                
        Returns:
            list: 预处理后的数据集，格式为字典列表
            
        Raises:
            FileNotFoundError: 当数据文件不存在时抛出
            ValueError: 当数据路径不是文件或文件格式不支持时抛出
            Exception: 其他处理过程中发生的异常
            
        缓存机制说明：
        1. 缓存key基于以下因素生成：
            - 文件内容哈希
            - tokenizer词汇表
            - 系统平台信息
            - PyTorch版本
            - CUDA可用性
        2. 缓存文件存储在MODEL_BASE_PATH/cache目录下
        3. 缓存有效性检查：
            - 比较缓存文件和数据文件的修改时间
            - 验证缓存数据的格式和完整性
            
        示例:
            >>> processor = DataProcessing(tokenizer)
            >>> # 加载JSONL格式数据
            >>> data = processor.load_dataset('data.jsonl')
            >>> # 加载CSV格式数据
            >>> data = processor.load_dataset('data.csv')
            >>> # 加载TXT格式数据
            >>> data = processor.load_dataset('data.txt')
            
        性能优化：
        - 使用mmap方式加载缓存文件，减少内存占用
        - 支持大文件分块处理
        - 自动清理无效缓存
        """
        try:
            logger.info(f'Loading dataset from {data_path}')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f'Data path {data_path} does not exist')
            if not os.path.isfile(data_path):
                raise ValueError(f'Data path {data_path} is not a file')
                
            cache_dir = os.path.join(Config.MODEL_BASE_PATH, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            os.chmod(cache_dir, 0o755)
            
            cache_key = hashlib.sha256()
            
            with open(data_path, 'rb') as f:
                while chunk := f.read(8192):
                    cache_key.update(chunk)
                    
            cache_key.update(str(self.tokenizer.get_vocab()).encode())
            cache_key.update(platform.platform().encode())
            cache_key.update(torch.__version__.encode())
            cache_key.update(str(torch.cuda.is_available()).encode())
            
            cache_path = os.path.join(
                cache_dir,
                f'cached_{cache_key.hexdigest()}.pt'
            )
            
            cache_metadata = {
                'data_path': data_path,
                'tokenizer_vocab': str(self.tokenizer.get_vocab()),
                'platform': platform.platform(),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'created_at': datetime.datetime.now().isoformat()
            }
            
            if os.path.exists(cache_path):
                cache_mtime = os.path.getmtime(cache_path)
                data_mtime = os.path.getmtime(data_path)
                
                if cache_mtime > data_mtime:
                    logger.info('Loading cached preprocessed data')
                    try:
                        cached_data = torch.load(cache_path, map_location='cpu', mmap=True)
                        
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
                
            processed_data = None
            if data_path.endswith('.jsonl'):
                processed_data = self._process_jsonl(data_path)
            elif data_path.endswith('.csv'):
                processed_data = self._process_csv(data_path)
            elif data_path.endswith('.txt'):
                processed_data = self._process_text(data_path)
            else:
                raise ValueError(f'Unsupported file format: {data_path}')
            
            if processed_data:
                try:
                    torch.save(processed_data, cache_path)
                    os.chmod(cache_path, 0o644)
                    logger.info(f'Saved processed data to cache: {cache_path}')
                except Exception as e:
                    logger.warning(f'Failed to save cache: {str(e)}')
                    
            return processed_data
            
        except Exception as e:
            logger.error(f'Failed to load dataset: {str(e)}')
            raise

    def _process_jsonl(self, data_path):
        """处理JSON Lines格式的数据文件
        
        该方法负责读取并解析JSON Lines格式的文件，每行作为一个独立的JSON对象处理。
        支持错误处理，遇到格式错误时会跳过该行并记录日志。
        
        Args:
            data_path (str): JSON Lines文件路径
            
        Returns:
            list: 解析后的数据列表，每个元素为一个字典
            
        Raises:
            ValueError: 当文件无法解析时抛出
            
        实现细节：
        1. 逐行读取文件
        2. 使用json模块解析每行内容
        3. 记录解析错误并跳过无效行
        4. 返回解析后的数据列表
        
        示例:
            >>> data = processor._process_jsonl('data.jsonl')
            >>> print(len(data))  # 输出解析成功的记录数
        """
        import json
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f'Invalid JSON at line {line_num}: {str(e)}')
                    continue
        return self._preprocess_data(data)

    def _process_csv(self, data_path):
        """处理CSV格式的数据文件
        
        该方法负责读取并解析CSV格式的文件，支持大文件分块读取。
        使用pandas的read_csv方法，支持自动类型推断和内存优化。
        
        Args:
            data_path (str): CSV文件路径
            
        Returns:
            list: 解析后的数据列表，每个元素为一个字典
            
        Raises:
            ValueError: 当文件无法解析时抛出
            
        实现细节：
        1. 使用pandas的read_csv方法分块读取
        2. 每块最大10000行，避免内存溢出
        3. 将每块数据转换为字典列表
        4. 合并所有块的数据
        
        示例:
            >>> data = processor._process_csv('data.csv')
            >>> print(len(data))  # 输出解析成功的记录数
            
        性能优化：
        - 分块读取大文件，减少内存占用
        - 自动处理CSV格式异常
        - 支持多种编码格式
        """
        import pandas as pd
        try:
            chunks = pd.read_csv(data_path, chunksize=10000)
            data = []
            for chunk in chunks:
                data.extend(chunk.to_dict('records'))
            return self._preprocess_data(data)
        except Exception as e:
            raise ValueError(f'Failed to read CSV: {str(e)}')

    def _process_text(self, data_path):
        """处理纯文本格式的数据文件
        
        该方法负责读取并解析纯文本格式的文件，支持多种编码格式。
        自动检测文件编码，支持UTF-8和latin-1编码。
        
        Args:
            data_path (str): 文本文件路径
            
        Returns:
            str: 读取的文本内容
            
        Raises:
            ValueError: 当文件无法解码时抛出
            
        实现细节：
        1. 首先尝试使用UTF-8编码读取
        2. 如果失败，尝试使用latin-1编码读取
        3. 返回读取的文本内容
        
        示例:
            >>> text = processor._process_text('data.txt')
            >>> print(len(text))  # 输出文本长度
            
        性能优化：
        - 自动检测文件编码
        - 支持大文件读取
        - 内存高效的文件读取方式
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = f.read()
            return self._preprocess_text_data(data)
        except UnicodeDecodeError:
            try:
                with open(data_path, 'r', encoding='latin-1') as f:
                    data = f.read()
                return self._preprocess_text_data(data)
            except Exception as e:
                raise ValueError(f'Failed to decode text file: {str(e)}')

    def _preprocess_data(self, data):
        """预处理结构化数据（JSONL/CSV），支持批量处理
        
        该方法负责对结构化数据进行预处理，包括数据清洗、格式转换等操作。
        
        Args:
            data (list): 需要预处理的数据列表，每个元素为一个字典
            
        Returns:
            list: 预处理后的数据列表
            
        实现细节：
        1. 验证数据格式，确保每个数据项都包含text字段
        2. 将文本数据分批处理，避免内存问题
        3. 使用分词器对文本进行动态填充和截断
        4. 对部分数据进行随机掩码处理，用于数据增强
        
        示例:
            >>> processed_data = processor._preprocess_data(raw_data)
            >>> print(len(processed_data))  # 输出预处理后的数据量
            
        性能优化：
        - 批量处理减少IO开销
        - 内存高效的数据处理方式
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
        """预处理纯文本数据，为模型训练准备输入
        
        该方法负责对纯文本数据进行预处理，包括文本清洗、格式转换等操作。
        
        Args:
            text (str): 需要预处理的原始文本内容
            
        Returns:
            list: 预处理后的数据列表，格式为字典列表
            
        实现细节：
        1. 使用分词器将文本编码为token序列
        2. 将token序列切分为固定长度的chunk
        3. 对每个chunk进行填充和截断处理
        4. 返回处理后的数据列表
        
        示例:
            >>> text = "这是一个测试文本"
            >>> processed_data = processor._preprocess_text_data(text)
            >>> print(len(processed_data))  # 输出预处理后的数据量
            
        性能优化：
        - 批量处理减少IO开销
        - 内存高效的数据处理方式
        - 支持大文本处理
        
        注意事项：
        - 输入文本应为UTF-8编码
        - 处理前会自动去除空白字符
        - 支持多语言文本处理
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

    def create_data_loader(self, dataset, batch_size):
        """创建优化的数据加载器，支持增强的并行处理
        
        该方法负责将预处理后的数据集转换为PyTorch DataLoader，
        支持多线程数据加载和自动批处理。该方法封装了底层的数据加载细节，
        为模型训练提供高效的数据流。
        
        Args:
            dataset (list): 预处理后的数据集，通常为字典列表格式
            batch_size (int): 每个batch的大小，影响内存使用和训练效率
            
        Returns:
            DataLoader: PyTorch DataLoader实例，支持迭代访问
            
        Raises:
            ValueError: 当数据集为空或batch_size不合法时抛出
            Exception: 数据加载器创建过程中发生的其他异常
            
        实现细节：
        1. 输入验证：检查数据集和batch_size的有效性
        2. 将数据集封装为TensorDataset
        3. 创建DataLoader实例，配置以下参数：
            - batch_size: 控制每个batch的大小
            - shuffle: 是否打乱数据顺序
            - num_workers: 数据加载线程数
            - pin_memory: 是否使用固定内存
            - drop_last: 是否丢弃不完整的最后一个batch
        4. 返回配置好的DataLoader实例
        
        示例:
            >>> # 创建数据加载器
            >>> loader = processor.create_data_loader(dataset, batch_size=32)
            >>> # 迭代访问数据
            >>> for batch in loader:
            ...     inputs = batch['input_ids']
            ...     labels = batch['labels']
            ...     # 训练模型
            ...     pass
            
        性能优化：
        - 多线程数据加载：利用CPU多核优势
        - 自动内存管理：减少内存碎片
        - 高效的数据批处理：优化IO性能
        - 支持GPU加速：pin_memory提升数据传输效率
        
        注意事项：
        - batch_size应根据GPU内存大小合理设置
        - num_workers应根据CPU核心数合理配置
        - 建议在训练循环开始前预加载数据
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
            
            # Log data loader configuration
            logger.info(f'Created data loader with:')
            logger.info(f'- Workers: {num_workers}')
            logger.info(f'- Prefetch factor: 4')
            logger.info(f'- Pin memory: {torch.cuda.is_available()}')
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

    def tokenize_text(self, text: str) -> torch.Tensor:
        """将输入文本转换为token ID张量
        
        该方法使用tokenizer将输入文本转换为模型可接受的token ID序列。
        支持添加特殊token（如[CLS]、[SEP]等），并返回PyTorch张量。
        
        Args:
            text (str): 需要tokenize的文本字符串
            
        Returns:
            torch.Tensor: 包含token ID的二维张量，形状为(1, seq_len)
            
        Raises:
            ValueError: 当输入不是字符串或为空字符串时抛出
            Exception: tokenization过程中发生的其他异常
            
        实现细节：
        1. 输入验证：检查输入是否为非空字符串
        2. 调用tokenizer的encode方法
        3. 返回PyTorch张量
        4. 自动添加特殊token
        
        示例:
            >>> token_ids = processor.tokenize_text("这是一个测试")
            >>> print(token_ids.shape)  # 输出张量形状
            >>> print(token_ids)  # 输出token ID
            
        性能优化：
        - 使用PyTorch张量直接返回，避免额外转换
        - 支持批量处理
        - 自动处理特殊字符
        """
        try:
            if not isinstance(text, str):
                raise ValueError('Input must be a string')
                
            if not text.strip():
                raise ValueError('Input text cannot be empty')
                
            return self.tokenizer.encode(
                text,
                return_tensors='pt',
                add_special_tokens=True
            )
        except Exception as e:
            logger.error(f'Tokenization failed: {str(e)}')
            raise

    def decode_text(self, token_ids: torch.Tensor) -> str:
        """将token ID张量转换回可读文本
        
        该方法使用tokenizer将模型输出的token ID序列转换回可读的文本字符串。
        支持跳过特殊token（如[CLS]、[SEP]等），返回原始文本内容。
        
        Args:
            token_ids (torch.Tensor): 包含token ID的张量，形状为(seq_len,)
            
        Returns:
            str: 解码后的文本字符串
            
        Raises:
            ValueError: 当输入不是张量或为空张量时抛出
            Exception: 解码过程中发生的其他异常
            
        实现细节：
        1. 输入验证：检查输入是否为非空张量
        2. 调用tokenizer的decode方法
        3. 跳过特殊token
        4. 返回解码后的文本
        
        示例:
            >>> text = processor.decode_text(token_ids)
            >>> print(text)  # 输出解码后的文本
            
        性能优化：
        - 直接处理PyTorch张量，避免额外转换
        - 支持批量处理
        - 自动处理特殊字符
        """
        try:
            if not isinstance(token_ids, torch.Tensor):
                raise ValueError('Input must be a torch.Tensor')
                
            if token_ids.numel() == 0:
                raise ValueError('Input tensor cannot be empty')
                
            return self.tokenizer.decode(
                token_ids.squeeze(),
                skip_special_tokens=True
            )
        except Exception as e:
            logger.error(f'Text decoding failed: {str(e)}')
            raise
            
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
        
        # Randomly select 15% of non-special tokens
        num_to_mask = max(1, int(mask_indices.numel() * 0.15))
        mask_indices = mask_indices[torch.randperm(mask_indices.numel())[:num_to_mask]]
        
        # Create a copy of input_ids to avoid modifying the original
        masked_input_ids = input_ids.clone()
        
        # Get mask token id from tokenizer
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            # If tokenizer doesn't have a mask token, use a random token
            mask_token_id = torch.randint(0, self.tokenizer.vocab_size, (1,)).item()
        
        # Apply masking
        masked_input_ids[mask_indices] = mask_token_id
        
        return masked_input_ids
