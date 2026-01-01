"""
Data handling utilities for transformer-based language models.

Provides PyTorch Dataset and DataLoader utilities for text classification
with transformer models that require tokenized inputs.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any
import numpy as np


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification with transformer models.
    
    Handles tokenization and creates tensors suitable for BERT-like models.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        tokenizer: Any = None,
        max_length: int = 256
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of input text strings
            labels: Optional array of labels (None for inference)
            tokenizer: Hugging Face tokenizer instance
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts for efficiency
        self.encodings = self._tokenize_all()
    
    def _tokenize_all(self) -> Dict[str, torch.Tensor]:
        """
        Tokenize all texts at once for efficiency.
        
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        return self.tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing input_ids, attention_mask, and optionally labels
        """
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class LMDataModule:
    """
    Data module for handling data loading and preparation for LM models.
    
    Provides a unified interface for creating DataLoaders from raw text data,
    compatible with the existing NLPDataLoader for data loading.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 256,
        batch_size: int = 16,
        num_workers: int = 0
    ):
        """
        Initialize the data module.
        
        Args:
            tokenizer: Hugging Face tokenizer instance
            max_length: Maximum sequence length
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for data loading (0 for Windows)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def create_dataloader(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = False,
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """
        Create a DataLoader from texts and optional labels.
        
        Args:
            texts: List of input texts
            labels: Optional array of labels
            shuffle: Whether to shuffle the data
            batch_size: Optional override for batch size
            
        Returns:
            PyTorch DataLoader
        """
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def prepare_data_from_loader(
        self,
        data_loader_output: Dict[str, Any]
    ) -> Dict[str, DataLoader]:
        """
        Prepare DataLoaders from NLPDataLoader output.
        
        This method bridges the existing data loading pipeline with
        transformer models by extracting raw texts and creating
        appropriate DataLoaders.
        
        Args:
            data_loader_output: Output from NLPDataLoader.prepare_data()
                               containing train_df, val_df, test_df
            
        Returns:
            Dictionary with train, val, test DataLoaders
        """
        result = {}
        
        # Process training data
        if 'train_df' in data_loader_output:
            train_texts = data_loader_output['train_df']['review'].tolist()
            train_labels = data_loader_output['y_train']
            result['train'] = self.create_dataloader(
                train_texts, train_labels, shuffle=True
            )
        
        # Process validation data
        if 'val_df' in data_loader_output:
            val_texts = data_loader_output['val_df']['review'].tolist()
            val_labels = data_loader_output['y_val']
            result['val'] = self.create_dataloader(
                val_texts, val_labels, shuffle=False
            )
        
        # Process test data
        if 'test_df' in data_loader_output:
            test_texts = data_loader_output['test_df']['review'].tolist()
            test_labels = data_loader_output['y_test']
            result['test'] = self.create_dataloader(
                test_texts, test_labels, shuffle=False
            )
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for dynamic batching.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary with stacked tensors
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
    
    return result



