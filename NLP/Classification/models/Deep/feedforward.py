"""
Feedforward Neural Network (MLP) models for text classification.

Simple but effective baseline deep learning models using:
- Word embeddings (averaged or concatenated)
- Multiple fully connected layers
- Dropout for regularization

Pure PyTorch implementation.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from models.Deep.base_deep import BaseDeepModel


class AveragingMLP(nn.Module):
    """
    MLP that averages word embeddings before classification.
    
    Architecture:
    1. Embedding layer
    2. Average pooling over sequence
    3. Multiple FC layers with ReLU and dropout
    4. Output layer
    
    Simple but surprisingly effective baseline.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize averaging MLP.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            lengths: Actual sequence lengths [batch]
            
        Returns:
            Logits [batch, num_classes]
        """
        # Embed tokens [batch, seq_len, embed_dim]
        embedded = self.embedding(input_ids)
        
        # Create mask for padding
        mask = (input_ids != 0).float().unsqueeze(-1)
        
        # Masked average pooling
        embedded_masked = embedded * mask
        summed = embedded_masked.sum(dim=1)
        
        # Avoid division by zero
        lengths_clamped = lengths.float().clamp(min=1).unsqueeze(-1)
        averaged = summed / lengths_clamped
        
        # MLP and output
        hidden = self.mlp(averaged)
        return self.output(hidden)


class DeepMLP(nn.Module):
    """
    Deeper MLP with more sophisticated architecture.
    
    Features:
    - Embedding with optional projection
    - Batch normalization
    - Residual connections
    - Multiple hidden layers
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize deep MLP.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes
            num_layers: Number of hidden layers (min 2)
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.use_batch_norm = use_batch_norm
        
        # Project embedding to hidden dim
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Build hidden layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        self.output = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Embed and pool
        embedded = self.embedding(input_ids)
        mask = (input_ids != 0).float().unsqueeze(-1)
        embedded_masked = embedded * mask
        summed = embedded_masked.sum(dim=1)
        lengths_clamped = lengths.float().clamp(min=1).unsqueeze(-1)
        pooled = summed / lengths_clamped
        
        # Project to hidden dimension
        hidden = self.projection(pooled)
        
        # Apply hidden layers with residual connections
        for i, layer in enumerate(self.layers):
            residual = hidden
            hidden = layer(hidden)
            if self.use_batch_norm:
                hidden = self.norms[i](hidden)
            hidden = torch.relu(hidden)
            hidden = self.dropouts[i](hidden)
            hidden = hidden + residual  # Residual connection
        
        return self.output(hidden)


class MLPClassifier(BaseDeepModel):
    """
    Simple MLP classifier using averaged word embeddings.
    
    Good baseline for text classification. Fast to train.
    
    Example:
        model = MLPClassifier(hidden_dim=256, num_layers=2)
        model.fit(train_texts, train_labels)
        predictions = model.predict(test_texts)
    """
    
    def __init__(
        self,
        num_layers: int = 2,
        **kwargs
    ):
        """
        Initialize MLP classifier.
        
        Args:
            num_layers: Number of hidden layers
            **kwargs: Arguments passed to BaseDeepModel
        """
        kwargs.setdefault('model_type', 'deep_mlp')
        super().__init__(**kwargs)
        self.num_layers = num_layers
    
    def _build_model(self) -> nn.Module:
        """Build the MLP model."""
        return AveragingMLP(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params['num_layers'] = self.num_layers
        return params


class DeepMLPClassifier(BaseDeepModel):
    """
    Deeper MLP classifier with batch norm and residual connections.
    
    More sophisticated than simple MLP, can capture more complex patterns.
    
    Example:
        model = DeepMLPClassifier(hidden_dim=512, num_layers=4)
        model.fit(train_texts, train_labels)
    """
    
    def __init__(
        self,
        num_layers: int = 3,
        use_batch_norm: bool = True,
        **kwargs
    ):
        """
        Initialize deep MLP classifier.
        
        Args:
            num_layers: Number of hidden layers
            use_batch_norm: Whether to use batch normalization
            **kwargs: Arguments passed to BaseDeepModel
        """
        kwargs.setdefault('model_type', 'deep_mlp_deep')
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
    
    def _build_model(self) -> nn.Module:
        """Build the deep MLP model."""
        return DeepMLP(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params['num_layers'] = self.num_layers
        params['use_batch_norm'] = self.use_batch_norm
        return params


