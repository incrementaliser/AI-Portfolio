"""
BERT and DistilBERT model implementations for text classification.

These models extend BaseLMModel to provide specific implementations
for BERT-family transformers with sensible defaults.
"""
from models.LMs.base_lm import BaseLMModel


class BERTClassifier(BaseLMModel):
    """
    BERT-based classifier for sentiment analysis.
    
    Uses bert-base-uncased by default. Fine-tuning is disabled by default (fine_tune=False).
    Set fine_tune=True to enable training on your dataset for better performance.
    Suitable for general-purpose text classification tasks.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        max_length: int = 256,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        fp16: bool = True,
        early_stopping_patience: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: Hugging Face model identifier (default: bert-base-uncased)
            num_labels: Number of classification labels (default: 2 for binary)
            max_length: Maximum sequence length (default: 256)
            batch_size: Training batch size (default: 16, reduce if OOM)
            learning_rate: Learning rate (default: 2e-5, standard for BERT)
            epochs: Training epochs (default: 3)
            warmup_ratio: Learning rate warmup fraction (default: 0.1)
            weight_decay: L2 regularization (default: 0.01)
            fp16: Mixed precision training (default: True)
            early_stopping_patience: Epochs before early stopping (default: 2)
            random_state: Random seed (default: 42)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            fp16=fp16,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
            **kwargs
        )
        
        # BERT-specific model type identifier
        self.model_type = "lm_bert"


class DistilBERTClassifier(BaseLMModel):
    """
    DistilBERT-based classifier for sentiment analysis.
    
    Uses distilbert-base-uncased by default. DistilBERT is a smaller, faster
    version of BERT that retains ~97% of BERT's performance while being
    60% faster and 40% smaller. Recommended for resource-constrained settings.
    
    Fine-tuning is disabled by default (fine_tune=False). Set fine_tune=True to enable training.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        max_length: int = 256,
        batch_size: int = 32,  # Can use larger batch due to smaller model
        learning_rate: float = 3e-5,  # Slightly higher LR works well
        epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        fp16: bool = True,
        early_stopping_patience: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize DistilBERT classifier.
        
        Args:
            model_name: Hugging Face model identifier (default: distilbert-base-uncased)
            num_labels: Number of classification labels (default: 2 for binary)
            max_length: Maximum sequence length (default: 256)
            batch_size: Training batch size (default: 32, larger due to smaller model)
            learning_rate: Learning rate (default: 3e-5, slightly higher than BERT)
            epochs: Training epochs (default: 3)
            warmup_ratio: Learning rate warmup fraction (default: 0.1)
            weight_decay: L2 regularization (default: 0.01)
            fp16: Mixed precision training (default: True)
            early_stopping_patience: Epochs before early stopping (default: 2)
            random_state: Random seed (default: 42)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            fp16=fp16,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
            **kwargs
        )
        
        # DistilBERT-specific model type identifier
        self.model_type = "lm_distilbert"


class RoBERTaClassifier(BaseLMModel):
    """
    RoBERTa-based classifier for sentiment analysis.
    
    Uses roberta-base by default. RoBERTa is an optimized version of BERT
    with improved pre-training procedure, often achieving better results.
    
    Fine-tuning is disabled by default (fine_tune=False). Set fine_tune=True to enable training.
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 2,
        max_length: int = 256,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        fp16: bool = True,
        early_stopping_patience: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize RoBERTa classifier.
        
        Args:
            model_name: Hugging Face model identifier (default: roberta-base)
            num_labels: Number of classification labels (default: 2 for binary)
            max_length: Maximum sequence length (default: 256)
            batch_size: Training batch size (default: 16)
            learning_rate: Learning rate (default: 2e-5)
            epochs: Training epochs (default: 3)
            warmup_ratio: Learning rate warmup fraction (default: 0.1)
            weight_decay: L2 regularization (default: 0.01)
            fp16: Mixed precision training (default: True)
            early_stopping_patience: Epochs before early stopping (default: 2)
            random_state: Random seed (default: 42)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            fp16=fp16,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
            **kwargs
        )
        
        # RoBERTa-specific model type identifier
        self.model_type = "lm_roberta"



