"""
Convolutional Neural Network (CNN) models for text classification.

1D CNNs are effective for capturing local n-gram patterns in text.
Based on the Kim (2014) CNN architecture.

Pure PyTorch implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import numpy as np

from models.Deep.base_deep import BaseDeepModel


class TextCNN(nn.Module):
    """
    CNN for text classification based on Kim (2014).
    
    Architecture:
    1. Embedding layer
    2. Multiple parallel conv1d layers with different kernel sizes
    3. Max pooling over time
    4. Concatenate and classify
    
    Captures n-gram patterns of different sizes (e.g., 2-gram, 3-gram, 4-gram).
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        num_filters: int = 100,
        filter_sizes: List[int] = [2, 3, 4, 5],
        dropout: float = 0.5
    ):
        """
        Initialize TextCNN.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_classes: Number of output classes
            num_filters: Number of filters per filter size
            filter_sizes: List of filter/kernel sizes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Create conv layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            lengths: Sequence lengths (not used here but required by interface)
            
        Returns:
            Logits [batch, num_classes]
        """
        # Embed: [batch, seq_len, embed_dim]
        embedded = self.embedding(input_ids)
        
        # Transpose for conv1d: [batch, embed_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions with ReLU and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv: [batch, num_filters, seq_len - kernel_size + 1]
            conv_out = F.relu(conv(embedded))
            # Max pool over time: [batch, num_filters]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs: [batch, num_filters * len(filter_sizes)]
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout and classify
        dropped = self.dropout(concatenated)
        return self.fc(dropped)


class DeepTextCNN(nn.Module):
    """
    Deeper CNN architecture with multiple conv layers.
    
    Features:
    - Stacked convolutional layers
    - Batch normalization
    - Residual connections (optional)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        num_filters: int = 128,
        filter_sizes: List[int] = [3, 4, 5],
        num_layers: int = 2,
        dropout: float = 0.5,
        use_residual: bool = True
    ):
        """
        Initialize deep TextCNN.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_classes: Number of output classes
            num_filters: Number of filters per conv layer
            filter_sizes: List of filter sizes for first layer
            num_layers: Number of stacked conv layers
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.use_residual = use_residual
        
        # First conv layer with multiple filter sizes
        self.first_convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs, padding=fs // 2)
            for fs in filter_sizes
        ])
        
        # Combined channels after first layer
        combined_filters = num_filters * len(filter_sizes)
        
        # Additional conv layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(combined_filters, combined_filters, 3, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm1d(combined_filters))
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(combined_filters, num_filters)
        self.fc2 = nn.Linear(num_filters, num_classes)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Embed and transpose
        embedded = self.embedding(input_ids).permute(0, 2, 1)
        
        # First conv layer (multiple filter sizes)
        first_outputs = [F.relu(conv(embedded)) for conv in self.first_convs]
        hidden = torch.cat(first_outputs, dim=1)  # [batch, combined, seq_len]
        
        # Additional conv layers with optional residual
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            residual = hidden if self.use_residual else None
            hidden = conv(hidden)
            hidden = bn(hidden)
            hidden = F.relu(hidden)
            if residual is not None:
                hidden = hidden + residual
        
        # Global max pooling
        pooled = F.adaptive_max_pool1d(hidden, 1).squeeze(2)
        
        # Classify
        out = self.dropout(pooled)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


class CNNClassifier(BaseDeepModel):
    """
    CNN classifier for text based on Kim (2014).
    
    Uses multiple parallel convolutional filters to capture
    n-gram patterns of different sizes.
    
    Example:
        model = CNNClassifier(num_filters=100, filter_sizes=[2,3,4,5])
        model.fit(train_texts, train_labels)
        predictions = model.predict(test_texts)
    """
    
    def __init__(
        self,
        num_filters: int = 100,
        filter_sizes: List[int] = None,
        **kwargs
    ):
        """
        Initialize CNN classifier.
        
        Args:
            num_filters: Number of filters per filter size
            filter_sizes: List of filter/kernel sizes (default: [2,3,4,5])
            **kwargs: Arguments passed to BaseDeepModel
        """
        kwargs.setdefault('model_type', 'deep_cnn')
        kwargs.setdefault('dropout', 0.5)  # Higher dropout for CNN
        super().__init__(**kwargs)
        
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes or [2, 3, 4, 5]
    
    def _build_model(self) -> nn.Module:
        """Build the CNN model."""
        return TextCNN(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            num_filters=self.num_filters,
            filter_sizes=self.filter_sizes,
            dropout=self.dropout
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params['num_filters'] = self.num_filters
        params['filter_sizes'] = self.filter_sizes
        return params


class DeepCNNClassifier(BaseDeepModel):
    """
    Deeper CNN classifier with stacked conv layers.
    
    More capacity than standard TextCNN, can learn more complex patterns.
    
    Example:
        model = DeepCNNClassifier(num_filters=128, num_layers=3)
        model.fit(train_texts, train_labels)
    """
    
    def __init__(
        self,
        num_filters: int = 128,
        filter_sizes: List[int] = None,
        num_layers: int = 2,
        use_residual: bool = True,
        **kwargs
    ):
        """
        Initialize deep CNN classifier.
        
        Args:
            num_filters: Number of filters per conv layer
            filter_sizes: Filter sizes for first layer
            num_layers: Number of stacked conv layers
            use_residual: Use residual connections
            **kwargs: Arguments passed to BaseDeepModel
        """
        kwargs.setdefault('model_type', 'deep_cnn_deep')
        kwargs.setdefault('dropout', 0.5)
        super().__init__(**kwargs)
        
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes or [3, 4, 5]
        self.num_layers = num_layers
        self.use_residual = use_residual
    
    def _build_model(self) -> nn.Module:
        """Build the deep CNN model."""
        return DeepTextCNN(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            num_filters=self.num_filters,
            filter_sizes=self.filter_sizes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_residual=self.use_residual
        )
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params['num_filters'] = self.num_filters
        params['filter_sizes'] = self.filter_sizes
        params['num_layers'] = self.num_layers
        params['use_residual'] = self.use_residual
        return params


