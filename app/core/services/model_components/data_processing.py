import os
import torch
import logging
import threading
import hashlib
import datetime
import psutil
import platform
from transformers import AutoTokenizer
from config import MODEL_BASE_PATH

logger = logging.getLogger(__name__)

class DataProcessing:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_dataset(self, data_path):
        """Load and preprocess dataset with enhanced caching and parallel processing"""
        try:
            logger.info(f'Loading dataset from {data_path}')
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f'Data path {data_path} does not exist')
            if not os.path.isfile(data_path):
                raise ValueError(f'Data path {data_path} is not a file')
                
            cache_dir = os.path.join(MODEL_BASE_PATH, 'cache')
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
        """Process JSON Lines format data"""
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
        """Process CSV format data"""
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
        """Process plain text format data"""
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
        """Preprocess structured data (JSONL/CSV) with batch processing"""
        from app.model.core_operations import ModelCoreOperations
        return ModelCoreOperations()._preprocess_data(data)

    def _preprocess_text_data(self, text):
        """Preprocess plain text data"""
        from app.model.core_operations import ModelCoreOperations
        return ModelCoreOperations()._preprocess_text_data(text)

    def create_data_loader(self, dataset, batch_size):
        """Create optimized data loader from dataset with enhanced parallel processing"""
        from app.model.core_operations import ModelCoreOperations
        return ModelCoreOperations().create_data_loader(dataset, batch_size)

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize input text into token IDs"""
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
        """Convert token IDs back to text"""
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
