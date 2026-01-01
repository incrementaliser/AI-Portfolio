"""
Attention-based models for text classification.

Includes:
- BiLSTM with self-attention
- BiLSTM with additive attention (Bahdanau style)
- BiLSTM with multi-head attention

Uses PyTorch Lightning for training.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

# PyTorch Lightning imports
LIGHTNING_AVAILABLE = False
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import lightning as pl
        from lightning.pytorch.callbacks import EarlyStopping
        LIGHTNING_AVAILABLE = True
    except ImportError:
        pass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseNLPModel
from models.Deep.base_deep import TextDataset, SimpleTokenizer


class SelfAttention(nn.Module):
    """
    Self-attention mechanism.
    
    Computes attention weights based on query-key similarity.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize self-attention.
        
        Args:
            hidden_dim: Dimension of hidden states
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        
        return attended, attn_weights


class AdditiveAttention(nn.Module):
    """
    Additive attention (Bahdanau attention).
    
    Computes attention using a learned alignment model.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        """
        Initialize additive attention.
        
        Args:
            hidden_dim: Dimension of hidden states
            attention_dim: Dimension of attention layer
        """
        super().__init__()
        self.attention = nn.Linear(hidden_dim, attention_dim)
        self.context = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply additive attention.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Compute attention scores
        attention_hidden = torch.tanh(self.attention(hidden_states))
        scores = self.context(attention_hidden).squeeze(-1)  # [batch, seq_len]
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq_len]
        
        # Compute weighted sum
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # [batch, 1, seq_len]
            hidden_states  # [batch, seq_len, hidden_dim]
        ).squeeze(1)  # [batch, hidden_dim]
        
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.
    
    Allows model to attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections and reshape to [batch, heads, seq, head_dim]
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores [batch, heads, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention [batch, heads, seq, head_dim]
        attended = torch.matmul(attn_weights, v)
        
        # Reshape and project [batch, seq, hidden_dim]
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output(attended)
        
        return output, attn_weights


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with attention for classification.
    
    Architecture:
    1. Embedding
    2. BiLSTM
    3. Attention mechanism (additive by default)
    4. FC classifier
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        attention_type: str = "additive",
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        """
        Initialize BiLSTM with attention.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_classes: Number of classes
            num_layers: Number of LSTM layers
            attention_type: Type of attention ("additive", "self", "multihead")
            num_heads: Number of heads (for multihead attention)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Choose attention type
        if attention_type == "additive":
            self.attention = AdditiveAttention(lstm_output_dim)
            self.use_pooling = True
        elif attention_type == "self":
            self.attention = SelfAttention(lstm_output_dim, dropout)
            self.use_pooling = False
        elif attention_type == "multihead":
            self.attention = MultiHeadAttention(lstm_output_dim, num_heads, dropout)
            self.use_pooling = False
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Embed
        embedded = self.embedding(input_ids)
        
        # Create mask
        mask = (input_ids != 0).float()
        
        # Pack sequences for LSTM
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sorted_embedded = embedded[sort_idx]
        sorted_lengths = sorted_lengths.clamp(min=1).cpu()
        
        packed = nn.utils.rnn.pack_padded_sequence(
            sorted_embedded, sorted_lengths, batch_first=True
        )
        
        # BiLSTM
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Pad to original length if needed
        batch_size, unpacked_len, hidden_dim = lstm_out.shape
        max_len = input_ids.shape[1]
        if unpacked_len < max_len:
            padding = torch.zeros(batch_size, max_len - unpacked_len, hidden_dim, device=lstm_out.device)
            lstm_out = torch.cat([lstm_out, padding], dim=1)
        
        # Unsort
        _, unsort_idx = sort_idx.sort()
        lstm_out = lstm_out[unsort_idx]
        
        # Apply attention
        if self.use_pooling:
            # Additive attention returns context vector directly
            context, _ = self.attention(lstm_out, mask)
        else:
            # Self/multihead attention returns attended sequence
            attended, _ = self.attention(lstm_out, mask)
            # Pool over sequence (mean pooling with mask)
            mask_expanded = mask.unsqueeze(-1)
            masked_attended = attended * mask_expanded
            context = masked_attended.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        # Classify
        context = self.dropout(context)
        return self.fc(context)


if LIGHTNING_AVAILABLE:
    class AttentionLightningModule(pl.LightningModule):
        """Lightning module for attention-based models."""
        
        def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
            """Initialize."""
            super().__init__()
            self.model = model
            self.learning_rate = learning_rate
            self.criterion = nn.CrossEntropyLoss()
            self.save_hyperparameters(ignore=['model'])
        
        def forward(self, input_ids, lengths):
            """Forward pass."""
            return self.model(input_ids, lengths)
        
        def training_step(self, batch, batch_idx):
            """Training step."""
            logits = self(batch['input_ids'], batch['length'])
            loss = self.criterion(logits, batch['labels'])
            acc = (logits.argmax(dim=1) == batch['labels']).float().mean()
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_acc', acc, prog_bar=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            """Validation step."""
            logits = self(batch['input_ids'], batch['length'])
            loss = self.criterion(logits, batch['labels'])
            acc = (logits.argmax(dim=1) == batch['labels']).float().mean()
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
        
        def configure_optimizers(self):
            """Configure optimizer."""
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class BiLSTMAttentionClassifier(BaseNLPModel):
    """
    Bidirectional LSTM with attention using PyTorch Lightning.
    
    Supports different attention mechanisms:
    - "additive": Bahdanau-style attention (default)
    - "self": Self-attention
    - "multihead": Multi-head attention
    
    Example:
        model = BiLSTMAttentionClassifier(
            hidden_dim=256,
            attention_type="additive"
        )
        model.fit(train_texts, train_labels)
        predictions = model.predict(test_texts)
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        attention_type: str = "additive",
        num_heads: int = 8,
        max_length: int = 256,
        dropout: float = 0.3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize BiLSTM with attention.
        
        Args:
            vocab_size: Maximum vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_classes: Number of classes
            num_layers: Number of LSTM layers
            attention_type: "additive", "self", or "multihead"
            num_heads: Number of attention heads
            max_length: Maximum sequence length
            dropout: Dropout probability
            batch_size: Batch size
            learning_rate: Learning rate
            epochs: Maximum epochs
            early_stopping_patience: Early stopping patience
            random_state: Random seed
            **kwargs: Additional arguments
        """
        model_type = f"deep_bilstm_{attention_type}_attention"
        super().__init__(model_type=model_type)
        
        if not LIGHTNING_AVAILABLE:
            raise ImportError("PyTorch Lightning required. pip install pytorch-lightning")
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.max_length = max_length
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        pl.seed_everything(random_state)
        
        self.model = None
        self.lightning_module = None
        self.tokenizer = None
        self._is_trained = False
    
    def _build_model(self) -> nn.Module:
        """Build attention model."""
        return BiLSTMAttention(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            attention_type=self.attention_type,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
    
    def _create_dataloader(self, texts, labels=None, shuffle=False):
        """Create DataLoader."""
        token_ids = self.tokenizer.encode_batch(texts)
        dataset = TextDataset(token_ids, labels, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)
    
    def _extract_texts(self, X):
        """Extract texts."""
        if hasattr(X, 'tolist'):
            return X.tolist()
        elif hasattr(X, 'values'):
            return X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        return list(X)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train model."""
        texts = self._extract_texts(X)
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        if self.tokenizer is None:
            self.tokenizer = SimpleTokenizer(max_vocab_size=self.vocab_size)
            self.tokenizer.fit(texts)
            print(f"  Vocabulary size: {self.tokenizer.vocab_size}")
        
        if self.model is None:
            self.model = self._build_model()
            self.lightning_module = AttentionLightningModule(self.model, self.learning_rate)
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"  Attention type: {self.attention_type}")
        
        train_loader = self._create_dataloader(texts, labels, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_texts = self._extract_texts(X_val)
            val_labels = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            val_loader = self._create_dataloader(val_texts, val_labels)
        
        callbacks = []
        if val_loader:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience, mode='min'))
        
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
            accelerator='auto',
            devices=1
        )
        
        print(f"\n  Training BiLSTM with {self.attention_type} attention...")
        trainer.fit(self.lightning_module, train_loader, val_loader)
        
        self._is_trained = True
        print("  Training completed!")
    
    def predict(self, X) -> np.ndarray:
        """Predict."""
        if not self.model:
            raise ValueError("Model not trained")
        
        texts = self._extract_texts(X)
        loader = self._create_dataloader(texts)
        
        self.model.eval()
        device = next(self.model.parameters()).device
        preds = []
        
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch['input_ids'].to(device), batch['length'].to(device))
                preds.extend(out.argmax(dim=1).cpu().numpy())
        
        return np.array(preds)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities."""
        if not self.model:
            raise ValueError("Model not trained")
        
        texts = self._extract_texts(X)
        loader = self._create_dataloader(texts)
        
        self.model.eval()
        device = next(self.model.parameters()).device
        probs = []
        
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch['input_ids'].to(device), batch['length'].to(device))
                probs.extend(F.softmax(out, dim=1).cpu().numpy())
        
        return np.array(probs)
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'attention_type': self.attention_type,
            'num_heads': self.num_heads,
            'max_length': self.max_length,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'random_state': self.random_state
        }


# Convenience aliases
BiLSTMAdditiveAttention = lambda **kw: BiLSTMAttentionClassifier(attention_type="additive", **kw)
BiLSTMSelfAttention = lambda **kw: BiLSTMAttentionClassifier(attention_type="self", **kw)
BiLSTMMultiHeadAttention = lambda **kw: BiLSTMAttentionClassifier(attention_type="multihead", **kw)


