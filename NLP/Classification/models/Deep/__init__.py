"""
Deep Learning models for text classification.

This module provides neural network architectures from simple to complex:

1. Feedforward Networks (MLP) - Simple baseline
2. Convolutional Networks (CNN) - Captures n-gram patterns
3. Recurrent Networks (LSTM/GRU) - Sequential modeling
4. Attention Networks (BiLSTM + Attention) - Focus on important tokens
5. Transformers - State-of-the-art architecture

Implementation mix:
- PyTorch: Feedforward, CNN, Transformer
- PyTorch Lightning: LSTM/GRU, Attention models

Example usage:
    from models.Deep import MLPClassifier, CNNClassifier, LSTMClassifier
    
    # Simple MLP baseline
    mlp = MLPClassifier(hidden_dim=256, num_layers=2)
    mlp.fit(train_texts, train_labels)
    
    # CNN for n-gram patterns
    cnn = CNNClassifier(num_filters=100, filter_sizes=[2,3,4,5])
    cnn.fit(train_texts, train_labels)
    
    # LSTM for sequences
    lstm = LSTMClassifier(hidden_dim=256, bidirectional=True)
    lstm.fit(train_texts, train_labels)
    
    # BiLSTM with attention
    attn = BiLSTMAttentionClassifier(attention_type="additive")
    attn.fit(train_texts, train_labels)
    
    # Custom Transformer
    transformer = TransformerClassifier(d_model=256, num_heads=8)
    transformer.fit(train_texts, train_labels)
"""

# Base classes
from models.Deep.base_deep import (
    BaseDeepModel,
    TextDataset,
    SimpleTokenizer
)

# Feedforward / MLP models (PyTorch)
from models.Deep.feedforward import (
    MLPClassifier,
    DeepMLPClassifier,
    AveragingMLP,
    DeepMLP
)

# CNN models (PyTorch)
from models.Deep.cnn import (
    CNNClassifier,
    DeepCNNClassifier,
    TextCNN,
    DeepTextCNN
)

# RNN models (PyTorch Lightning)
LIGHTNING_AVAILABLE = False
try:
    from models.Deep.rnn import (
        LSTMClassifier,
        GRUClassifier,
        BiLSTMClassifier,
        BiGRUClassifier,
        LSTMModule,
        GRUModule
    )
    LIGHTNING_AVAILABLE = True
except ImportError:
    pass

# Attention models (PyTorch Lightning)
ATTENTION_AVAILABLE = False
try:
    from models.Deep.attention import (
        BiLSTMAttentionClassifier,
        BiLSTMAttention,
        SelfAttention,
        AdditiveAttention,
        MultiHeadAttention
    )
    ATTENTION_AVAILABLE = True
except ImportError:
    pass

# Transformer models (PyTorch)
from models.Deep.transformer import (
    TransformerClassifier,
    TransformerTiny,
    TransformerSmall,
    TransformerBase,
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionalEncoding
)

__all__ = [
    # Base
    'BaseDeepModel',
    'TextDataset',
    'SimpleTokenizer',
    
    # Feedforward
    'MLPClassifier',
    'DeepMLPClassifier',
    'AveragingMLP',
    'DeepMLP',
    
    # CNN
    'CNNClassifier',
    'DeepCNNClassifier',
    'TextCNN',
    'DeepTextCNN',
    
    # Transformer
    'TransformerClassifier',
    'TransformerTiny',
    'TransformerSmall',
    'TransformerBase',
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'PositionalEncoding',
    
    # Availability flags
    'LIGHTNING_AVAILABLE',
    'ATTENTION_AVAILABLE',
]

# Add RNN exports if Lightning available
if LIGHTNING_AVAILABLE:
    __all__.extend([
        'LSTMClassifier',
        'GRUClassifier',
        'BiLSTMClassifier',
        'BiGRUClassifier',
        'LSTMModule',
        'GRUModule',
    ])

# Add attention exports if available
if ATTENTION_AVAILABLE:
    __all__.extend([
        'BiLSTMAttentionClassifier',
        'BiLSTMAttention',
        'SelfAttention',
        'AdditiveAttention',
        'MultiHeadAttention',
    ])


