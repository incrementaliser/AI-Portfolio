"""
Custom Transformer model for text classification.

Implements a Transformer encoder from scratch for educational purposes.
Based on "Attention Is All You Need" (Vaswani et al., 2017).

Pure PyTorch implementation.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseNLPModel
from models.Deep.base_deep import BaseDeepModel, TextDataset, SimpleTokenizer


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    
    Adds position information to embeddings using sine and cosine functions.
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Position-encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    
    Consists of:
    1. Multi-head self-attention
    2. Position-wise feedforward network
    3. Layer normalization and residual connections
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feedforward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual and norm
        attn_output, _ = self.self_attention(
            x, x, x,
            key_padding_mask=(mask == 0) if mask is not None else None
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual and norm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Full Transformer encoder for text classification.
    
    Architecture:
    1. Token embedding
    2. Positional encoding
    3. Stack of Transformer encoder layers
    4. Pooling (CLS token or mean)
    5. Classification head
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_length: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        pooling: str = "mean"
    ):
        """
        Initialize Transformer encoder.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feedforward dimension
            max_length: Maximum sequence length
            num_classes: Number of output classes
            dropout: Dropout probability
            pooling: Pooling strategy ("mean", "cls", "max")
        """
        super().__init__()
        
        self.d_model = d_model
        self.pooling = pooling
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        # Optional CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            lengths: Sequence lengths [batch]
            
        Returns:
            Logits [batch, num_classes]
        """
        # Create attention mask
        mask = (input_ids != 0).float()
        
        # Embed and scale
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add CLS token if using CLS pooling
        if self.pooling == "cls":
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            # Extend mask for CLS token
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Pool to single vector
        if self.pooling == "cls":
            pooled = x[:, 0]  # CLS token
        elif self.pooling == "mean":
            # Mean pooling with mask
            mask_expanded = mask.unsqueeze(-1)
            x_masked = x * mask_expanded
            pooled = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            # Max pooling
            x_masked = x.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            pooled, _ = x_masked.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        return self.classifier(pooled)


class TransformerClassifier(BaseDeepModel):
    """
    Custom Transformer classifier for text.
    
    A from-scratch implementation of Transformer encoder
    for text classification.
    
    Example:
        model = TransformerClassifier(
            d_model=256,
            num_heads=8,
            num_layers=4
        )
        model.fit(train_texts, train_labels)
        predictions = model.predict(test_texts)
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        num_classes: int = 2,
        max_length: int = 256,
        pooling: str = "mean",
        dropout: float = 0.1,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        epochs: int = 10,
        early_stopping_patience: int = 3,
        warmup_steps: int = 100,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize Transformer classifier.
        
        Args:
            vocab_size: Maximum vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feedforward dimension
            num_classes: Number of output classes
            max_length: Maximum sequence length
            pooling: Pooling strategy ("mean", "cls", "max")
            dropout: Dropout probability
            batch_size: Training batch size
            learning_rate: Learning rate
            epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
            warmup_steps: Number of warmup steps for scheduler
            random_state: Random seed
            **kwargs: Additional arguments
        """
        # Don't call super().__init__() - we'll set things up manually
        BaseNLPModel.__init__(self, model_type="deep_transformer")
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.max_length = max_length
        self.pooling = pooling
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.warmup_steps = warmup_steps
        self.random_state = random_state
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.model = None
        self.tokenizer = None
        self._is_trained = False
    
    def _build_model(self) -> nn.Module:
        """Build the Transformer model."""
        return TransformerEncoder(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            max_length=self.max_length,
            num_classes=self.num_classes,
            dropout=self.dropout,
            pooling=self.pooling
        )
    
    def _create_dataloader(self, texts, labels=None, shuffle=False):
        """Create DataLoader."""
        token_ids = self.tokenizer.encode_batch(texts)
        dataset = TextDataset(token_ids, labels, self.max_length)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )
    
    def _extract_texts(self, X):
        """Extract texts."""
        if hasattr(X, 'tolist'):
            return X.tolist()
        elif hasattr(X, 'values'):
            return X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        return list(X)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the Transformer model.
        
        Uses custom training loop with warmup scheduler.
        """
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
            print(f"  Transformer config: d_model={self.d_model}, heads={self.num_heads}, layers={self.num_layers}")
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create dataloaders
        train_loader = self._create_dataloader(texts, labels, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_texts = self._extract_texts(X_val)
            val_labels = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            val_loader = self._create_dataloader(val_texts, val_labels, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.01
        )
        
        # Warmup + cosine decay scheduler
        total_steps = len(train_loader) * self.epochs
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        global_step = 0
        
        print(f"\n  Training Transformer for up to {self.epochs} epochs...")
        print(f"  Device: {self.device}")
        print(f"  Warmup steps: {self.warmup_steps}, Total steps: {total_steps}")
        
        from tqdm import tqdm
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            progress = tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{self.epochs}")
            
            for batch in progress:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                lengths = batch['length'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, lengths)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            avg_train_loss = total_loss / num_batches
            print(f"  Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                print(f"  Epoch {epoch + 1} - Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        if best_model_state:
                            self.model.load_state_dict(best_model_state)
                            self.model.to(self.device)
                        break
        
        self._is_trained = True
        print("  Training completed!")
    
    def _evaluate(self, dataloader, criterion):
        """Evaluate model."""
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
                preds = outputs.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def predict(self, X) -> np.ndarray:
        """Predict labels."""
        if not self.model:
            raise ValueError("Model not trained")
        
        texts = self._extract_texts(X)
        loader = self._create_dataloader(texts)
        
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                lengths = batch['length'].to(self.device)
                out = self.model(input_ids, lengths)
                preds.extend(out.argmax(dim=-1).cpu().numpy())
        
        return np.array(preds)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities."""
        if not self.model:
            raise ValueError("Model not trained")
        
        texts = self._extract_texts(X)
        loader = self._create_dataloader(texts)
        
        self.model.eval()
        probs = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                lengths = batch['length'].to(self.device)
                out = self.model(input_ids, lengths)
                probs.extend(F.softmax(out, dim=-1).cpu().numpy())
        
        return np.array(probs)
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'pooling': self.pooling,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'warmup_steps': self.warmup_steps,
            'random_state': self.random_state,
            'device': str(self.device)
        }


# Convenience aliases for different sizes
class TransformerTiny(TransformerClassifier):
    """Tiny Transformer (fast training)."""
    
    def __init__(self, **kwargs):
        """Initialize tiny transformer."""
        kwargs.setdefault('d_model', 128)
        kwargs.setdefault('num_heads', 4)
        kwargs.setdefault('num_layers', 2)
        kwargs.setdefault('d_ff', 512)
        super().__init__(**kwargs)


class TransformerSmall(TransformerClassifier):
    """Small Transformer (balanced)."""
    
    def __init__(self, **kwargs):
        """Initialize small transformer."""
        kwargs.setdefault('d_model', 256)
        kwargs.setdefault('num_heads', 8)
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('d_ff', 1024)
        super().__init__(**kwargs)


class TransformerBase(TransformerClassifier):
    """Base Transformer (higher capacity)."""
    
    def __init__(self, **kwargs):
        """Initialize base transformer."""
        kwargs.setdefault('d_model', 512)
        kwargs.setdefault('num_heads', 8)
        kwargs.setdefault('num_layers', 6)
        kwargs.setdefault('d_ff', 2048)
        super().__init__(**kwargs)


