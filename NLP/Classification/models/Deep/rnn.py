"""
Recurrent Neural Network (RNN) models for text classification.

Includes LSTM and GRU variants, both unidirectional and bidirectional.
Uses PyTorch Lightning for cleaner training code.

PyTorch Lightning implementation.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

# PyTorch Lightning imports
LIGHTNING_AVAILABLE = False
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import lightning as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        LIGHTNING_AVAILABLE = True
    except ImportError:
        pass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseNLPModel
from models.Deep.base_deep import TextDataset, SimpleTokenizer


class LSTMModule(nn.Module):
    """
    LSTM model for text classification.
    
    Architecture:
    1. Embedding layer
    2. LSTM (uni/bidirectional)
    3. Take final hidden state(s)
    4. Fully connected classifier
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM module.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_classes: Number of output classes
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout probability
        """
        super().__init__()
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Embed
        embedded = self.embedding(input_ids)
        
        # Pack for efficient LSTM
        # Sort by length for packing
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sorted_embedded = embedded[sort_idx]
        
        # Clamp lengths to avoid empty sequences
        sorted_lengths = sorted_lengths.clamp(min=1).cpu()
        
        packed = nn.utils.rnn.pack_padded_sequence(
            sorted_embedded, sorted_lengths, batch_first=True
        )
        
        # LSTM forward
        _, (hidden, _) = self.lstm(packed)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Unsort to restore original order
        _, unsort_idx = sort_idx.sort()
        hidden = hidden[unsort_idx]
        
        # Classify
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class GRUModule(nn.Module):
    """
    GRU model for text classification.
    
    Similar to LSTM but with simpler gating mechanism.
    Often faster and similarly effective.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.3
    ):
        """Initialize GRU module."""
        super().__init__()
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        embedded = self.embedding(input_ids)
        
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sorted_embedded = embedded[sort_idx]
        sorted_lengths = sorted_lengths.clamp(min=1).cpu()
        
        packed = nn.utils.rnn.pack_padded_sequence(
            sorted_embedded, sorted_lengths, batch_first=True
        )
        
        _, hidden = self.gru(packed)
        
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        _, unsort_idx = sort_idx.sort()
        hidden = hidden[unsort_idx]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)


if LIGHTNING_AVAILABLE:
    class RNNLightningModule(pl.LightningModule):
        """
        PyTorch Lightning module for RNN models.
        
        Wraps LSTM or GRU for training with Lightning.
        """
        
        def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 1e-3,
            num_classes: int = 2
        ):
            """
            Initialize Lightning module.
            
            Args:
                model: The RNN model (LSTM or GRU)
                learning_rate: Learning rate
                num_classes: Number of classes
            """
            super().__init__()
            self.model = model
            self.learning_rate = learning_rate
            self.criterion = nn.CrossEntropyLoss()
            self.save_hyperparameters(ignore=['model'])
        
        def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return self.model(input_ids, lengths)
        
        def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
            """Training step."""
            logits = self(batch['input_ids'], batch['length'])
            loss = self.criterion(logits, batch['labels'])
            
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch['labels']).float().mean()
            
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_acc', acc, prog_bar=True)
            return loss
        
        def validation_step(self, batch: Dict, batch_idx: int) -> None:
            """Validation step."""
            logits = self(batch['input_ids'], batch['length'])
            loss = self.criterion(logits, batch['labels'])
            
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch['labels']).float().mean()
            
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
        
        def configure_optimizers(self):
            """Configure optimizer."""
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class BaseRNNClassifier(BaseNLPModel):
    """
    Base class for RNN-based classifiers using PyTorch Lightning.
    
    Provides common functionality for LSTM and GRU models.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        bidirectional: bool = False,
        max_length: int = 256,
        dropout: float = 0.3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        random_state: int = 42,
        model_type: str = "deep_rnn",
        **kwargs
    ):
        """
        Initialize RNN classifier.
        
        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of RNN hidden state
            num_classes: Number of output classes
            num_layers: Number of RNN layers
            bidirectional: Use bidirectional RNN
            max_length: Maximum sequence length
            dropout: Dropout probability
            batch_size: Training batch size
            learning_rate: Learning rate
            epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            random_state: Random seed
            model_type: Type identifier
            **kwargs: Additional arguments
        """
        super().__init__(model_type=model_type)
        
        if not LIGHTNING_AVAILABLE:
            raise ImportError(
                "PyTorch Lightning is required for RNN classifiers. "
                "Install with: pip install pytorch-lightning"
            )
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.max_length = max_length
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        pl.seed_everything(random_state)
        
        # Model components
        self.model = None
        self.lightning_module = None
        self.tokenizer = None
        self.trainer = None
        self._is_trained = False
    
    def _build_rnn_module(self) -> nn.Module:
        """Build the RNN module. Override in subclasses."""
        raise NotImplementedError
    
    def _create_dataloader(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """Create DataLoader."""
        token_ids = self.tokenizer.encode_batch(texts)
        dataset = TextDataset(token_ids, labels, self.max_length)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def _extract_texts(self, X) -> List[str]:
        """Extract texts from various formats."""
        if hasattr(X, 'tolist'):
            return X.tolist()
        elif hasattr(X, 'values'):
            return X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        return list(X)
    
    def fit(
        self,
        X,
        y: np.ndarray,
        X_val=None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """Train the model using PyTorch Lightning."""
        texts = self._extract_texts(X)
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Build tokenizer
        if self.tokenizer is None:
            self.tokenizer = SimpleTokenizer(max_vocab_size=self.vocab_size)
            self.tokenizer.fit(texts)
            print(f"  Vocabulary size: {self.tokenizer.vocab_size}")
        
        # Build model
        if self.model is None:
            self.model = self._build_rnn_module()
            self.lightning_module = RNNLightningModule(
                self.model, 
                self.learning_rate,
                self.num_classes
            )
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create dataloaders
        train_loader = self._create_dataloader(texts, labels, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_texts = self._extract_texts(X_val)
            val_labels = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            val_loader = self._create_dataloader(val_texts, val_labels, shuffle=False)
        
        # Setup callbacks
        callbacks = []
        if val_loader is not None:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    mode='min'
                )
            )
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
            accelerator='auto',
            devices=1,
            deterministic=True
        )
        
        print(f"\n  Training with PyTorch Lightning for up to {self.epochs} epochs...")
        
        # Train
        self.trainer.fit(
            self.lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        self._is_trained = True
        print("  Training completed!")
    
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        texts = self._extract_texts(X)
        dataloader = self._create_dataloader(texts, labels=None, shuffle=False)
        
        self.model.eval()
        device = next(self.model.parameters()).device
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                lengths = batch['length'].to(device)
                
                outputs = self.model(input_ids, lengths)
                preds = torch.argmax(outputs, dim=-1)
                all_predictions.extend(preds.cpu().numpy())
        
        return np.array(all_predictions)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        texts = self._extract_texts(X)
        dataloader = self._create_dataloader(texts, labels=None, shuffle=False)
        
        self.model.eval()
        device = next(self.model.parameters()).device
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                lengths = batch['length'].to(device)
                
                outputs = self.model(input_ids, lengths)
                probs = torch.softmax(outputs, dim=-1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'max_length': self.max_length,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'random_state': self.random_state
        }


class LSTMClassifier(BaseRNNClassifier):
    """
    LSTM classifier for text using PyTorch Lightning.
    
    Example:
        model = LSTMClassifier(hidden_dim=256, num_layers=2)
        model.fit(train_texts, train_labels, val_texts, val_labels)
        predictions = model.predict(test_texts)
    """
    
    def __init__(self, **kwargs):
        """Initialize LSTM classifier."""
        kwargs.setdefault('model_type', 'deep_lstm')
        super().__init__(**kwargs)
    
    def _build_rnn_module(self) -> nn.Module:
        """Build LSTM module."""
        return LSTMModule(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )


class GRUClassifier(BaseRNNClassifier):
    """
    GRU classifier for text using PyTorch Lightning.
    
    Similar to LSTM but faster training.
    
    Example:
        model = GRUClassifier(hidden_dim=256, bidirectional=True)
        model.fit(train_texts, train_labels)
    """
    
    def __init__(self, **kwargs):
        """Initialize GRU classifier."""
        kwargs.setdefault('model_type', 'deep_gru')
        super().__init__(**kwargs)
    
    def _build_rnn_module(self) -> nn.Module:
        """Build GRU module."""
        return GRUModule(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )


class BiLSTMClassifier(LSTMClassifier):
    """
    Bidirectional LSTM classifier.
    
    Captures context from both directions.
    """
    
    def __init__(self, **kwargs):
        """Initialize BiLSTM classifier."""
        kwargs['bidirectional'] = True
        kwargs.setdefault('model_type', 'deep_bilstm')
        super().__init__(**kwargs)


class BiGRUClassifier(GRUClassifier):
    """
    Bidirectional GRU classifier.
    
    Fast bidirectional model.
    """
    
    def __init__(self, **kwargs):
        """Initialize BiGRU classifier."""
        kwargs['bidirectional'] = True
        kwargs.setdefault('model_type', 'deep_bigru')
        super().__init__(**kwargs)


