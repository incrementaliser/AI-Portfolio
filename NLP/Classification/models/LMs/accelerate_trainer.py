"""
Accelerate-based trainer for transformer models.

Uses HuggingFace Accelerate for:
- Multi-GPU training
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Hardware-agnostic code

Based on recommendations from suggestions.txt for Phase 1 (Traditional/BERT).
"""
import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Accelerate import
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseNLPModel


class AccelerateTrainer(BaseNLPModel):
    """
    Transformer trainer using HuggingFace Accelerate.
    
    Provides hardware-agnostic training that automatically handles:
    - Single GPU, multi-GPU, and CPU training
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Distributed training (if configured)
    
    Example usage:
        trainer = AccelerateTrainer(model_name="bert-base-uncased")
        trainer.fit(train_texts, train_labels, val_texts, val_labels)
        predictions = trainer.predict(test_texts)
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
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",  # "no", "fp16", "bf16"
        early_stopping_patience: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the Accelerate-based trainer.
        
        Args:
            model_name: Hugging Face model identifier
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            batch_size: Per-device batch size
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            warmup_ratio: Fraction of steps for LR warmup
            weight_decay: Weight decay for AdamW
            gradient_accumulation_steps: Steps to accumulate gradients
            mixed_precision: Precision mode ("no", "fp16", "bf16")
            early_stopping_patience: Epochs before early stopping
            random_state: Random seed
            **kwargs: Additional arguments
        """
        super().__init__(model_type="lm_accelerate")
        
        if not ACCELERATE_AVAILABLE:
            raise ImportError(
                "Accelerate is required for AccelerateTrainer. "
                "Install with: pip install accelerate"
            )
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision if torch.cuda.is_available() else "no"
        )
        
        # Model components
        self.tokenizer = None
        self.model = None
        self._is_trained = False
        
        print(f"  Accelerate initialized:")
        print(f"    Device: {self.accelerator.device}")
        print(f"    Mixed Precision: {self.accelerator.mixed_precision}")
        print(f"    Num Processes: {self.accelerator.num_processes}")
    
    def _load_model(self) -> None:
        """Load pretrained model and tokenizer."""
        print(f"  Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def _create_dataloader(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """Create a DataLoader from texts and labels."""
        from models.LMs.lm_data import TextClassificationDataset
        
        dataset = TextClassificationDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True
        )
    
    def fit(
        self,
        X: Any,
        y: np.ndarray,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Train the model using Accelerate.
        
        Args:
            X: Training texts
            y: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        if self.model is None:
            self._load_model()
        
        # Process input texts
        texts = self._extract_texts(X)
        labels = np.array(y) if not isinstance(y, np.ndarray) else y
        
        # Create dataloaders
        train_loader = self._create_dataloader(texts, labels, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_texts = self._extract_texts(X_val)
            val_labels = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            val_loader = self._create_dataloader(val_texts, val_labels, shuffle=False)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Calculate total steps
        num_update_steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
        total_steps = num_update_steps_per_epoch * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare with Accelerator (handles device placement, distributed, etc.)
        self.model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, scheduler
        )
        
        if val_loader is not None:
            val_loader = self.accelerator.prepare(val_loader)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n  Training with Accelerate for {self.epochs} epochs...")
        print(f"  Effective batch size: {self.batch_size * self.gradient_accumulation_steps * self.accelerator.num_processes}")
        print(f"  Total optimization steps: {total_steps}")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_steps = 0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"  Epoch {epoch + 1}/{self.epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
                    
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                num_steps += 1
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / num_steps
            
            if self.accelerator.is_main_process:
                print(f"  Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                
                if self.accelerator.is_main_process:
                    print(f"  Epoch {epoch + 1} - Validation loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.accelerator.wait_for_everyone()
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    self._best_model_state = {
                        k: v.cpu().clone() for k, v in unwrapped.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if self.accelerator.is_main_process:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        # Restore best model
                        if hasattr(self, '_best_model_state'):
                            unwrapped = self.accelerator.unwrap_model(self.model)
                            unwrapped.load_state_dict(self._best_model_state)
                        break
        
        self._is_trained = True
        if self.accelerator.is_main_process:
            print("  Training completed!")
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Calculate validation loss."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                total_loss += outputs.loss.item()
                num_batches += 1
        
        # Gather losses across processes
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _extract_texts(self, X: Any) -> List[str]:
        """Extract text list from various input formats."""
        if hasattr(X, 'tolist'):
            return X.tolist()
        elif hasattr(X, 'values'):
            return X.values.tolist() if hasattr(X.values, 'tolist') else list(X.values)
        return list(X)
    
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if hasattr(X, 'toarray'):
            raise ValueError("LM models require raw text input.")
        
        texts = self._extract_texts(X)
        dataloader = self._create_dataloader(texts, labels=None, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                predictions = torch.argmax(outputs.logits, dim=-1)
                gathered = self.accelerator.gather(predictions)
                all_predictions.extend(gathered.cpu().numpy())
        
        return np.array(all_predictions[:len(texts)])
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if hasattr(X, 'toarray'):
            raise ValueError("LM models require raw text input.")
        
        texts = self._extract_texts(X)
        dataloader = self._create_dataloader(texts, labels=None, shuffle=False)
        dataloader = self.accelerator.prepare(dataloader)
        
        self.model.eval()
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                gathered = self.accelerator.gather(probs)
                all_probabilities.extend(gathered.cpu().numpy())
        
        return np.array(all_probabilities[:len(texts)])
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'warmup_ratio': self.warmup_ratio,
            'weight_decay': self.weight_decay,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'mixed_precision': self.mixed_precision,
            'early_stopping_patience': self.early_stopping_patience,
            'random_state': self.random_state,
            'device': str(self.accelerator.device)
        }
    
    def save(self, path: str) -> None:
        """Save the model."""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            import json
            config = self.get_params()
            config['_is_trained'] = self._is_trained
            with open(os.path.join(path, 'training_config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"  Model saved to: {path}")
    
    def load(self, path: str) -> None:
        """Load a saved model."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        
        # Prepare with accelerator
        self.model = self.accelerator.prepare(self.model)
        
        import json
        config_path = os.path.join(path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self._is_trained = config.get('_is_trained', True)
        else:
            self._is_trained = True
        
        print(f"  Model loaded from: {path}")


# Convenience class aliases for different model types
class BERTAccelerate(AccelerateTrainer):
    """BERT classifier using Accelerate for training."""
    
    def __init__(self, **kwargs):
        """Initialize BERT with Accelerate."""
        kwargs.setdefault('model_name', 'bert-base-uncased')
        super().__init__(**kwargs)


class DistilBERTAccelerate(AccelerateTrainer):
    """DistilBERT classifier using Accelerate for training."""
    
    def __init__(self, **kwargs):
        """Initialize DistilBERT with Accelerate."""
        kwargs.setdefault('model_name', 'distilbert-base-uncased')
        super().__init__(**kwargs)


class RoBERTaAccelerate(AccelerateTrainer):
    """RoBERTa classifier using Accelerate for training."""
    
    def __init__(self, **kwargs):
        """Initialize RoBERTa with Accelerate."""
        kwargs.setdefault('model_name', 'roberta-base')
        super().__init__(**kwargs)


