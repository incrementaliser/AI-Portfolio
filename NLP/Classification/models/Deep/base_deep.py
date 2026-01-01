"""
Base class for deep learning models.

Provides common functionality for PyTorch-based deep learning models
including embedding handling, training loops, and evaluation.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseNLPModel


class TextDataset(Dataset):
    """
    PyTorch Dataset for text classification with token indices.
    
    Handles conversion of tokenized text to tensors.
    """
    
    def __init__(
        self,
        texts: List[List[int]],
        labels: Optional[np.ndarray] = None,
        max_length: int = 256
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of tokenized texts (list of token indices)
            labels: Optional array of labels
            max_length: Maximum sequence length (for padding/truncation)
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.texts[idx]
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        item = {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'length': torch.tensor(min(len(self.texts[idx]), self.max_length), dtype=torch.long)
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class SimpleTokenizer:
    """
    Simple word-level tokenizer for deep learning models.
    
    Builds vocabulary from training data and converts text to indices.
    """
    
    def __init__(
        self,
        max_vocab_size: int = 30000,
        min_freq: int = 2,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>"
    ):
        """
        Initialize tokenizer.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a word to be included
            pad_token: Padding token
            unk_token: Unknown token
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.word_freq = {}
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> 'SimpleTokenizer':
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Self for chaining
        """
        # Count word frequencies
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Sort by frequency and add to vocabulary
        sorted_words = sorted(
            self.word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        idx = len(self.word2idx)
        for word, freq in sorted_words:
            if freq < self.min_freq:
                break
            if idx >= self.max_vocab_size:
                break
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        self._is_fitted = True
        return self
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of token indices
        """
        words = text.lower().split()
        return [self.word2idx.get(word, 1) for word in words]  # 1 = UNK
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Convert batch of texts to token indices.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of token index lists
        """
        return [self.encode(text) for text in texts]
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)
    
    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'max_vocab_size': self.max_vocab_size,
                'min_freq': self.min_freq
            }, f)
    
    def load(self, path: str) -> None:
        """Load tokenizer from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.max_vocab_size = data['max_vocab_size']
        self.min_freq = data['min_freq']
        self._is_fitted = True


class BaseDeepModel(BaseNLPModel):
    """
    Base class for PyTorch deep learning models.
    
    Provides common functionality including:
    - Tokenization and vocabulary building
    - Training loop with validation
    - Early stopping
    - Model saving/loading
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        max_length: int = 256,
        dropout: float = 0.3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        random_state: int = 42,
        model_type: str = "deep",
        **kwargs
    ):
        """
        Initialize base deep model.
        
        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes
            max_length: Maximum sequence length
            dropout: Dropout probability
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            random_state: Random seed
            model_type: Type identifier for the model
            **kwargs: Additional arguments
        """
        super().__init__(model_type=model_type)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_length = max_length
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components (initialized in subclasses)
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[SimpleTokenizer] = None
        self._is_trained = False
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build the neural network architecture.
        
        Must be implemented by subclasses.
        
        Returns:
            PyTorch nn.Module
        """
        pass
    
    def _create_dataloader(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Create DataLoader from texts.
        
        Args:
            texts: List of text strings
            labels: Optional labels
            shuffle: Whether to shuffle
            
        Returns:
            PyTorch DataLoader
        """
        # Tokenize texts
        token_ids = self.tokenizer.encode_batch(texts)
        
        dataset = TextDataset(token_ids, labels, self.max_length)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )
    
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Train the model.
        
        Args:
            X: Training texts
            y: Training labels
            X_val: Validation texts
            y_val: Validation labels
        """
        # Extract texts
        texts = self._extract_texts(X)
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Build tokenizer
        if self.tokenizer is None:
            self.tokenizer = SimpleTokenizer(max_vocab_size=self.vocab_size)
            self.tokenizer.fit(texts)
            print(f"  Vocabulary size: {self.tokenizer.vocab_size}")
        
        # Build model
        if self.model is None:
            self.model = self._build_model()
            self.model.to(self.device)
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create dataloaders
        train_loader = self._create_dataloader(texts, labels, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_texts = self._extract_texts(X_val)
            val_labels = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            val_loader = self._create_dataloader(val_texts, val_labels, shuffle=False)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\n  Training for up to {self.epochs} epochs...")
        print(f"  Device: {self.device}")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch + 1}/{self.epochs}",
                leave=True
            )
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                lengths = batch['length'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, lengths)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / num_batches
            print(f"  Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                print(f"  Epoch {epoch + 1} - Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        if best_model_state is not None:
                            self.model.load_state_dict(best_model_state)
                            self.model.to(self.device)
                        break
        
        self._is_trained = True
        print("  Training completed!")
    
    def _evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Evaluate model on a dataloader.
        
        Args:
            dataloader: DataLoader to evaluate on
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                lengths = batch['length'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, lengths)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def _extract_texts(self, X: Any) -> List[str]:
        """Extract text list from various input formats."""
        if hasattr(X, 'tolist'):
            return X.tolist()
        elif hasattr(X, 'values'):
            return X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        return list(X)
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input texts
            
        Returns:
            Array of predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if hasattr(X, 'toarray'):
            raise ValueError("Deep models require raw text input.")
        
        texts = self._extract_texts(X)
        dataloader = self._create_dataloader(texts, labels=None, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                outputs = self.model(input_ids, lengths)
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
        
        return np.array(all_predictions)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input texts
            
        Returns:
            Array of shape (n_samples, num_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if hasattr(X, 'toarray'):
            raise ValueError("Deep models require raw text input.")
        
        texts = self._extract_texts(X)
        dataloader = self._create_dataloader(texts, labels=None, shuffle=False)
        
        self.model.eval()
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                outputs = self.model(input_ids, lengths)
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                all_probabilities.extend(probs.cpu().numpy())
        
        return np.array(all_probabilities)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'random_state': self.random_state,
            'device': str(self.device)
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save tokenizer
        self.tokenizer.save(os.path.join(path, 'tokenizer.json'))
        
        # Save config
        import json
        config = self.get_params()
        config['_is_trained'] = self._is_trained
        config['actual_vocab_size'] = self.tokenizer.vocab_size
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  Model saved to: {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        import json
        
        # Load config
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self._is_trained = config.get('_is_trained', True)
        
        # Load tokenizer
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load(os.path.join(path, 'tokenizer.json'))
        
        # Build and load model
        self.vocab_size = config.get('actual_vocab_size', self.tokenizer.vocab_size)
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        self.model.to(self.device)
        
        print(f"  Model loaded from: {path}")


